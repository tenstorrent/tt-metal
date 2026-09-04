# 05 — Decision log (append-only)

One block per judgement call, numbered monotonically from `DEC-001`. A superseded entry is never
edited — a new entry says `Supersedes DEC-NNN`. Template: recipe §1.3.

A `DEC` is mandatory when you: pick a number not read verbatim from `config.json`; choose between
repo patterns; deviate from the recipe; lower a threshold or skip a case; add an env var; leave
something stubbed; or find that the reference and the repo disagree.

---

### DEC-001 — Which checkpoint is `llama31_8b`?
- **Phase / module:** P0 / model card
- **Date (UTC):** 2026-09-04
- **Trigger:** `llama31_8b_d_p` is a directory name; the card cannot be filled until the checkpoint
  behind it is pinned. Recipe P0 step 2 requires a `DEC` naming the resolved id and the method.
- **Question:** which exact checkpoint do the card's dims describe, and how is that established
  without trusting the directory name?
- **Options considered:**
  1. Trust `$HF_MODEL` and read its `config.json`. Cheap; establishes *what is staged*, not *what it
     is*.
  2. Diff `$HF_MODEL/config.json` against the repo's own vendored
     `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json`. Establishes identity
     against an in-repo, independently-maintained copy, and needs no network.
  3. Download `meta-llama/Llama-3.1-8B-Instruct/config.json` from the gated HF repo and diff. The
     strongest link to the public id, but adds a network + token dependency to a P0 fact.
- **Choice:** option 2 (and option 1 as its input). The resolved identity is
  `meta-llama/Llama-3.1-8B-Instruct`, staged at `/home/mstojkovic/models/Llama-3.1-8B-Instruct`.
- **Why:** the two files are **byte-identical**, not merely dim-compatible
  (`md5 = 3cd5831d379b509d53afade0e24c36e9` for both), so the card's dims are confirmed against a
  second, in-tree source with zero network dependency. The recipe (§The machine) states the live
  gated repo also matches; that is inherited as a verified machine fact, not re-derived.
- **Evidence:** `md5sum /home/mstojkovic/models/Llama-3.1-8B-Instruct/config.json
  models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json` → same digest;
  `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json:13` (`"hidden_act": "silu"`).
  The config is bundled verbatim at
  `models/demos/llama31_8b_d_p/configs/Llama-3.1-8B-Instruct/config.json`.
- **Confidence:** high.
- **Falsifier:** the byte-identity check failing after a checkpoint re-stage, or an HF `config.json`
  for that repo id differing from the bundled copy.
- **Revisit if:** the checkpoint is re-staged, or the target is re-pointed at another Llama-3.x 8B
  variant (base rather than Instruct would change nothing dimensional, but the card's identity row
  would be wrong).
- **Blast radius:** the whole card, every dimension in every module, `G-CARD`, and the bundled
  `configs/` copy that dimension-only tests read.

---

### DEC-002 — Package skeleton: reconcile P0 step 1 with the P3 tree
- **Phase / module:** P0 / package skeleton
- **Date (UTC):** 2026-09-04
- **Trigger:** the recipe specifies the skeleton twice and the two do not agree.
  P0 step 1 (`BRINGUP_RECIPE.md:485`) says create `{tt,tests/unit,scripts,docs}` *with `__init__.py`
  files*; the P3 target tree (`BRINGUP_RECIPE.md:752`) contains no `docs/` and no
  `scripts/__init__.py`, and both template packages (`models/demos/minimax_m3/scripts`,
  `models/demos/gpt_oss_d_p/scripts`) ship `scripts/` without an `__init__.py`.
- **Question:** which of the two spellings does P0 create?
- **Options considered:**
  1. Literal P0: create `docs/` and `scripts/__init__.py`. Follows the phase actually being
     executed; risks two files that P9's "no dead files" gate must then justify.
  2. Literal P3: no `docs/`, no `scripts/__init__.py`. Matches both templates; silently disobeys the
     step being executed.
- **Choice:** option 1 — the recipe's own instruction for *this* phase wins — with the conflict
  logged here so P9 can delete rather than rediscover.
- **Why:** the rule is "where your instincts differ from the recipe, follow the recipe"; the phase
  under execution is the more specific instruction. An empty `docs/` costs nothing now and is a
  one-line deletion at P9 if it is still empty. `docs/.gitkeep` exists only because git does not
  track empty directories.
- **Also decided here (same trigger, same blast radius):**
  - **No `reference/` package** — explicit in P0 step 1 and P1 (Llama loads in HF `transformers`
    with no `trust_remote_code`). Not a deviation; recorded so its absence is not read as an omission.
  - **The recipe is not vendored into the package.** The P3 tree lists
    `models/demos/llama31_8b_d_p/BRINGUP_RECIPE.md`, but the recipe for this bring-up lives in the
    kit at `models/demos/common/bringup/BRINGUP_RECIPE.md` and this session may not modify anything
    outside the package; copying it in would fork it. `scripts/verify_citations.py`'s
    `DOC_PREFIXES["BRINGUP_RECIPE.md"]` is repointed at the kit copy so the shorthand still resolves,
    and the kit copy is scanned by the doc pass. Revisit at P3/P9.
- **Evidence:** `BRINGUP_RECIPE.md:485`, `BRINGUP_RECIPE.md:752`; `models/demos/minimax_m3/tests/test_factory.py:14`
  (`from ..config import MeshConfig` — M3 keeps `config.py` at the package root, another shape the
  P3 tree does not use).
- **Confidence:** medium — this is a coin-flip between two recipe statements, not a technical call.
- **Falsifier:** P9's `G-CLEAN` rejecting an empty `docs/` or an unused `scripts/__init__.py`.
- **Revisit if:** P3 writes the outline (it must state which spelling it commits to), or P9 runs.
- **Blast radius:** two files; `G-CARD`, `G-OUTLINE`, `G-CLEAN`.

---

### DEC-003 — Adapting the kit's `verify_citations.py` to this package
- **Phase / module:** P0 / `scripts/verify_citations.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the kit example copied verbatim (with only `PKG` set, as its docstring instructs)
  **fails its own doc scan** on a fresh package.
- **Question:** what is the minimum change that makes the verifier truthful here?
- **What actually happened:** the example's own explanatory comment contains backticked
  package-relative refs (tt/config.py line 134, tests/unit/test_reference_model.py line 136) as *prose
  examples*. Pass 2 globs `scripts/*.py` — so it scans itself — and, because this package has no
  `tt/config.py` until P5, resolves the bare `tt/config.py` onto
  `models/demos/gpt_oss_d_p/tt/attention/config.py` (108 lines) and reports `DOC OUT OF RANGE`. That
  is precisely the citation-shadowing false positive the comment is describing, occurring inside the
  comment describing it.
- **Options considered:**
  1. Leave it failing until P5 creates `tt/config.py`. Would make `G-CARD`'s "citations clean"
     unachievable in P0 for a reason unrelated to P0.
  2. Remove the backticks from those two prose refs (keep the words). They are illustrations, not
     claims, so nothing verifiable is lost.
  3. Teach the resolver to ignore refs inside comments. A parser for a problem this small.
- **Choice:** option 2, plus `DOC_PREFIXES["BRINGUP_RECIPE.md"]` → the kit path (`DEC-002`), plus
  populating `CITES` with the 53 P0–P2 claims and `DOCS` with the nine log files + the recipe.
- **Why:** keeps the verifier's exit code meaningful from the first phase. The change is confined to
  comment text.
- **Evidence:** first run of the unmodified copy: `DOC OUT OF RANGE ... gpt_oss_d_p/tt/attention/config.py:134-134
  (file has 108 lines)`. The same run also caught **six wrong line numbers in this agent's own first
  draft of `CITES`** (the `sdpa_device_operation.cpp` `TT_FATAL`, the `PREFILL_MIGRATION_TESTING.md`
  shape constraints, `hidden_act` in the vendored config, `models/demos/minimax_m3/conftest.py`'s session scope,
  `models/demos/gpt_oss_d_p/utils/substate.py`'s `def substate`) — which is the argument for the tool.
- **Confidence:** high.
- **Falsifier:** a real wrong citation slipping through because its backticks were removed. Only the
  two prose refs in the copied comment were touched; both name files this package does not have yet.
- **Revisit if:** P5 creates `tt/config.py` (the refs could be re-backticked then).
- **Blast radius:** `scripts/verify_citations.py`; every doc gate.

---

### DEC-004 — `CHUNK_SIZE` / `MAX_SEQ_LEN`: record the constraint in P0, pick the number in P7
- **Phase / module:** P0 / model card §4
- **Date (UTC):** 2026-09-04
- **Trigger:** recipe P0 step 5 requires the chunk/`max_seq_len` constraints in the card, but the
  values are only exercised by the chunked runtime (P7) and the engine (P10).
- **Question:** pin concrete values now, or record the constraint and defer?
- **Options considered:**
  1. Pin now (e.g. `CHUNK_SIZE = 1024`, `MAX_SEQ_LEN = 4096`). A number chosen before anything
     measures it — and the recipe forbids picking numbers that later get refitted.
  2. Record the *constraint* in the card (P0's actual requirement is the arithmetic) and pick the
     value in P7, where `G-CHUNK` measures it. Appendix B warns that
     `max_seq_len == chunk_size` silently measures the SP bootstrap instead of the cache-read path,
     which is a P7-side consideration.
- **Choice:** option 2. Recorded constraint at SP=4: `CHUNK_SIZE % 128 == 0`,
  `MAX_SEQ_LEN % CHUNK_SIZE == 0`, and (from Appendix B) `MAX_SEQ_LEN > CHUNK_SIZE` so more than one
  chunk actually runs.
- **Why:** P0's requirement is the arithmetic, and the recipe's own rule is to set a number before
  the measurement it will be compared to — not before the *phase* that gives it meaning.
- **Evidence:** `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md:62`
  ("Two shape constraints apply: `MAX_SEQ_LEN % CHUNK_SIZE == 0` and `CHUNK_SIZE % (SP*32) == 0`");
  `BRINGUP_RECIPE.md:1834` (the "too good" symptom when `max_seq_len == chunk_size`).
- **Confidence:** high.
- **Falsifier:** P7 discovering a constraint that P0 should have carried (e.g. a chunk-size ceiling
  from DRAM capacity at 32 layers), which would make the deferral a mistake rather than an ordering.
- **Revisit if:** P7 starts, or the engine manifest in P10 pins a different default
  (`PREFILL_CHUNK_SIZE` defaults to `5*1024`).
- **Blast radius:** `07_RISKS.md` R-004, P7 (`G-CHUNK`), P10 (`G-REQUEST`).

---

### DEC-005 — Reference strategy: in-test torch math + HF for layer/model level
- **Phase / module:** P1 / reference
- **Date (UTC):** 2026-09-04
- **Trigger:** P1 requires the reference strategy to be chosen and logged before any module test is
  written.
- **Question:** what is the oracle for the per-module gates, and what for the layer/model gates?
- **Options considered:**
  1. HF `transformers` `LlamaForCausalLM` everywhere, via the `models/tt_transformers`
     `reference_*` accessors (`models/tt_transformers/tt/model_config.py:4037`, `:4393`, `:4410`, `:4365`, `:4167`,
     `:4379`, `:4027`). Nothing to vendor; but every call needs `HF_MODEL`
     (`models/tt_transformers/tt/model_config.py:702` raises without it), loads a 15 GB checkpoint, and drags
     `ModelArgs` into the inner loop.
  2. A vendored `reference/model.py` (the `minimax_m3` / `gpt_oss_d_p` pattern). Explicitly ruled
     out by the recipe for Llama, and would need keeping in sync with HF.
  3. Hand-written torch math inside each unit test, both sides driven by identical random weights
     (`models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py`). Fast, no checkpoint, runs on a bare card — but it
     is a transcription, so it can be faithfully wrong.
- **Choice:** the recipe's combination — **3 for the P5 module gates, 1 for the P6/P7 layer- and
  model-level gates**, with `G-REF` cross-validating 3 against 1 on one layer.
- **Why:** option 3 keeps the inner loop checkpoint-free and fast, which is what makes a module gate
  cheap enough to run on every edit; option 1 is the only oracle that can be *wrong about the
  architecture* in the same way HF is, which is what the layer/model gates need. `G-REF` is what
  stops 3 drifting from 1 — and its bit-exactness must be read honestly: it proves the transcription
  is faithful, not that either side is right about Llama.
- **Evidence:** `models/tt_transformers/tests/test_mlp.py` (canonical use of the accessors);
  `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:12` (identical-random-weights pattern);
  `models/tt_transformers/tt/model_config.py:702` (the `HF_MODEL` hard requirement).
- **Confidence:** high.
- **Falsifier:** `G-REF` failing to reach PCC ≥ 0.9999 between the two oracles on one layer — which
  would mean the transcription is not faithful and option 3 cannot stand in for option 1.
- **Revisit if:** a module's hand-written reference becomes longer than the module it checks, or HF
  changes `LlamaAttention`'s forward signature (P1 trap 5).
- **Blast radius:** every `tests/unit/test_*_vs_ref.py`, `G-REF`, `G-RMS`, `G-ROPE`, `G-MLP`,
  `G-ATTN`, `G-LAYER`, `G-MODEL`.

---

### DEC-006 — Reference dtype policy: fp32 everywhere, quantise only what the device stores
- **Phase / module:** P1 / reference
- **Date (UTC):** 2026-09-04
- **Trigger:** the checkpoint's `torch_dtype` is `bfloat16`, so the *default* way to build a torch
  reference (`from_pretrained`) produces a bf16-weight reference.
- **Question:** at what precision is the reference computed, and what is quantised?
- **Options considered:**
  1. Load/compute the reference at the checkpoint dtype (bf16). What HF does by default, and what
     `models/tt_transformers/tests/test_rms_norm.py`'s oracle does.
  2. fp32 reference; quantise **only the tensors the device stores** (weights, inputs) via
     `quantize_like_device`, and do all remaining arithmetic in fp32.
- **Choice:** option 2, for both oracles, with the cast happening only at the comparison boundary.
- **Why:** a bf16-weight reference shares the device's own rounding and inflates the PCC — measured
  in the recipe as 0.9999867 (bf16-weight reference) vs 0.99995 (fp32-weight reference) for the
  *same device output*. Two incomparable numbers, and the flattering one is the default. The fp32
  reference is strictly harder, and it is the only one the noise-floor method (§2.2) is defined
  against: the floor *is* "quantise what is stored, compute in fp32".
- **Evidence:** `BRINGUP_RECIPE.md:320` (§2.1(a), the measured pair);
  `models/demos/common/bringup/examples/noise_floor.py:33` (`quantize_like_device`, host-only, so it is a pure
  quantiser and never a compute path).
- **Confidence:** high.
- **Falsifier:** a gate where the fp32-reference PCC is *higher* than the bf16-reference PCC on the
  same device output — which would mean the quantisation is not modelling what the device stores.
- **Revisit if:** a module stores something in a dtype `quantize_like_device` cannot express (it
  requires a 4D tile-shaped tensor), or a fused kernel makes the floor meaningless (§2.3).
- **Blast radius:** every numeric gate; `tests/test_factory.py`'s floor helpers.

---

### DEC-007 — One definition of the floor helpers, copied not imported
- **Phase / module:** P1 / `tests/test_factory.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `quantize_like_device` / `err_ratio` exist in the kit at
  `models/demos/common/bringup/examples/noise_floor.py`; the recipe (§2.2) says they must have exactly one
  definition, in `tests/test_factory.py`, and rule 4 says "reuse means *import*, not copy-paste".
- **Question:** import them from the kit example, or define them in `tests/test_factory.py`?
- **Options considered:**
  1. `from models.demos.common.bringup.examples.noise_floor import ...`. Literal reuse — but the kit
     is documentation: `examples/` is not an importable package (no `__init__.py`), its README calls
     it a thing to "copy into your package's test helpers", and a package depending on a doc
     directory would break the moment the kit is moved or versioned.
  2. Copy both functions into `tests/test_factory.py` as the package's single definition.
- **Choice:** option 2 — copy, with the kit cited as the source in the docstring.
- **Why:** this is the copy-paste that rule 4 requires a `DEC` for, and the justification is that the
  source is an example file, not a library: `models/demos/common/bringup/examples/` has no `__init__.py`, and
  `models/demos/common/bringup/README.md:20` describes `noise_floor.py` as "The threshold primitive. Copy it
  into your package's test helpers."
- **Evidence:** `models/demos/common/bringup/README.md:20`; `models/demos/common/bringup/examples/noise_floor.py:1-60`;
  recipe §2.2 "Keep one definition of the floor helpers … two copies drift".
- **Confidence:** high.
- **Falsifier:** the kit gaining a real importable package for these helpers, which would make the
  copy the wrong call.
- **Revisit if:** a second copy of either helper appears anywhere in the package (P9 should grep for
  it), or the kit is packaged.
- **Blast radius:** `tests/test_factory.py`, every numeric gate's floor.

---

### DEC-008 — `TestFactory.setup_test` is deferred to P5, not written blind in P1
- **Phase / module:** P1 / `tests/test_factory.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** P1 step 2 requires `TestFactory.setup_test(mesh_device, ...)` to build
  `MeshConfig` + `CCLManager` — but `tt/config.py` and `tt/ccl.py` are **P5.1** deliverables and do
  not exist in P1. The template it points at,
  `models/demos/minimax_m3/tests/test_factory.py:14`, imports both plus `ModelArgs`.
- **Question:** write a `setup_test` against modules that do not exist, or leave it out of P1?
- **Options considered:**
  1. Write it now with imports inside the function body, so `tests/test_factory.py` still imports
     cleanly. It would be untested, unexercised, and would have to be rewritten once the real
     constructor signatures exist — a stub, which the recipe requires a `DEC` for anyway.
  2. Ship the P1 half of `test_factory.py` (`llama_config_dims`, `requires_hf_reference`,
     `hf_model_path`, the floor helpers) and add `setup_test` in P5.1 in the same edit that creates
     `MeshConfig`/`CCLManager`.
- **Choice:** option 2.
- **Why:** the P1 gate (`G-REF`) needs none of `setup_test`; nothing in P1–P4 can call it; and a stub
  written against guessed signatures is exactly the dead code `G-CLEAN` rejects. This is a **recipe
  ordering defect**, not a judgement call about the design — P1 step 2 asks for a device-side factory
  four phases before its dependencies exist.
- **Evidence:** `BRINGUP_RECIPE.md:648-655` (P1 step 2) vs `BRINGUP_RECIPE.md:1011` (P5.1 creates
  `tt/config.py` + `tt/ccl.py`); `models/demos/minimax_m3/tests/test_factory.py:14-17` (the template's imports).
- **Confidence:** high.
- **Falsifier:** a P2–P4 deliverable turning out to need `setup_test`.
- **Revisit if:** P5.1 starts — it must add `setup_test` in the same edit as `MeshConfig`.
- **Blast radius:** `tests/test_factory.py`; P5's unit tests; `G-MESH`.

---

### DEC-009 — Where the bundled-config byte-identity assertion lives
- **Phase / module:** P1 / `tests/unit/test_reference_model.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** P0 step 2 says "Bundle the resolved `config.json` verbatim in the package and assert
  byte-identity in a test" — but P0 creates no test file, and the P3 tree has no test that owns this
  assertion.
- **Question:** which file owns the assertion?
- **Options considered:**
  1. A new `tests/unit/test_config_bundle.py`. Clear ownership, but adds a file the P3 tree does not
     list, and the recipe says every gate owns something *in that tree*.
  2. Put it in `tests/unit/test_reference_model.py` (`G-REF`'s owner, created in P1). The bundled
     config is the reference's dimension source, so the placement is coherent.
- **Choice:** option 2 — `test_bundled_config_matches_checkpoint`, skipped by
  `requires_hf_reference` when no checkpoint is staged (the assertion is meaningless without one).
- **Why:** keeps the tree exactly as P3 specifies while still executing the P0 requirement, one
  phase later than the step that states it. `G-CARD` records the same fact from the `md5sum`
  transcript, so P0's verdict does not depend on a P1 file.
- **Evidence:** `BRINGUP_RECIPE.md:496` (the requirement); `BRINGUP_RECIPE.md:752-830` (the tree, which
  has no config-bundle test).
- **Confidence:** high.
- **Falsifier:** a reviewer looking for the assertion under a name containing "config" and not
  finding it. Mitigated: it is named in `01_REFERENCE.md` and in `G-REF`'s gate block.
- **Blast radius:** `tests/unit/test_reference_model.py`, `G-CARD`, `G-REF`.

---

### DEC-010 — `tests/unit/test_reference_model.py` runs on **host only**, no `mesh_device`
- **Phase / module:** P1 / `tests/unit/test_reference_model.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the phase map bills P1 as "Device: host", but the file it points at as the pattern,
  `models/demos/minimax_m3/tests/unit/test_reference_model.py`, takes the `mesh_device` fixture.
- **Question:** does `G-REF` open a device?
- **Options considered:**
  1. Mirror the M3 file and take `mesh_device`. Costs a device open (~20 s here) for a test whose
     assertions are all torch-side, and makes `G-REF` unrunnable on a machine with no card.
  2. Host-only: no `mesh_device`, no `ttnn` import at module scope.
- **Choice:** option 2.
- **Why:** everything `G-REF` proves — determinism, hand-written-vs-HF agreement, causality, the
  `rope_theta` trap, the bundled-config identity — is torch and JSON. The phase map's "host" is the
  authority; the M3 file needs a device because *its* reference is a device-adjacent wrapper.
  Keeping ttnn out also means `G-REF` still runs when the box is busy with a mesh job.
- **Evidence:** `BRINGUP_RECIPE.md:46` (P1 row, Device = host);
  `models/demos/minimax_m3/tests/unit/test_reference_model.py` (uses `mesh_device`).
- **Confidence:** high.
- **Falsifier:** a `G-REF` sub-check that turns out to need a device (none does today).
- **Blast radius:** `tests/unit/test_reference_model.py`, `G-REF`.

---

### DEC-011 — Hand-written reference: HF *convention* (`rotate_half`), not Meta interleaved
- **Phase / module:** P1 / `tests/unit/test_reference_model.py` and every P5 module test
- **Date (UTC):** 2026-09-04
- **Trigger:** RoPE has two conventions in this tree and the reference must commit to one before any
  module test is written; `G-REF` compares the hand-written oracle against HF.
- **Question:** does the hand-written torch reference implement HF's `rotate_half` (halves) or Meta's
  interleaved layout?
- **Options considered:**
  1. HF convention. Directly comparable to `LlamaAttention` (which is what `G-REF` cross-checks),
     and it is what `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:94` does.
  2. Meta interleaved. Matches what the *device* wants
     (`ttnn.experimental.rotary_embedding_llama` + `reverse_permute`d weights), so a device test
     could skip a conversion.
- **Choice:** option 1 for the reference; the device side gets the Meta-format weights and cos/sin
  via `convert_hf_qkv_to_meta_format` (`models/tt_transformers/tt/load_checkpoints.py:451`) and the
  interleaved table, exactly as the gpt-oss test does.
- **Why:** the reference's job is to agree with HF; the swizzle is the *device* path's problem and
  belongs in the module's weight loader, where a mismatch is a loud PCC failure on `G-ROPE` rather
  than a silently wrong oracle. Appendix B lists "RoPE convention mismatch" as the top cause of
  attention PCC 0.5–0.9, and the mitigation is one frequency set feeding both tables — which this
  arrangement enforces.
- **Evidence:** `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:94-100` (one `inv_freq`, two tables);
  `models/tt_transformers/tt/load_checkpoints.py:451`, `:891` (`reverse_permute`).
- **Confidence:** high.
- **Falsifier:** `G-ROPE`'s device-vs-reference PCC collapsing while the reference alone agrees with
  HF bit-exactly — that would mean the swizzle, not the convention choice, is wrong.
- **Revisit if:** P5.3 finds `rotary_embedding_llama` needs a different table layout than the
  gpt-oss test builds.
- **Blast radius:** `tests/unit/test_reference_model.py`, P5.3 `tt/rope.py`, `G-ROPE`, `G-ATTN`.

---

### DEC-012 — Checkpoint loading lives in `tests/test_factory.py` until `ModelArgs` exists
- **Phase / module:** P1 / `conftest.py`, `tests/test_factory.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** P1 step 3 says copy the `state_dict` fixture's shape from
  `models/demos/minimax_m3/conftest.py`, whose body is
  `ModelArgs.load_state_dict(model_path, dummy_weights=False)`
  (`models/demos/minimax_m3/conftest.py:22`). `ModelArgs` is a **P6.2** deliverable here.
- **Question:** where does the loader live in P1?
- **Options considered:**
  1. Write a minimal `tt/model_config.py` now just to host `load_state_dict`. Pulls a P6 file four
     phases forward, and it would be rewritten when the real `ModelArgs` lands.
  2. Put a `load_hf_state_dict(prefixes=None, model_path=None)` in `tests/test_factory.py` and have
     `conftest.py` call it. Test-side code, where it belongs while the only consumers are tests.
  3. Load the whole 15 GB checkpoint in `G-REF`. Would make a host-only gate take minutes.
- **Choice:** option 2, with a `prefixes` argument so a test can read just the tensors it needs
  (`G-REF` reads only `model.layers.0.` — 9 tensors — via the safetensors index, and the whole file
  runs in 13.6 s).
- **Why:** keeps `tt/` empty until P5, keeps the gate fast, and gives P6.2 a single call site to move
  when `ModelArgs` arrives.
- **Evidence:** `models/demos/minimax_m3/conftest.py:22`; `models/demos/llama31_8b_d_p/tests/test_factory.py`
  `load_hf_state_dict`; `G-REF` wall clock 13.6 s for 10 tests including two real-weight ones.
- **Confidence:** high.
- **Falsifier:** a P5 device test needing a state-dict shape this loader cannot produce (it returns
  the raw HF keys; the HF→Meta mapping is `models/tt_transformers/tt/load_checkpoints.py:800`'s job,
  not the loader's).
- **Revisit if:** P6.2 creates `ModelArgs` — the fixture body moves there and this helper becomes a
  thin wrapper or is deleted.
- **Blast radius:** `conftest.py`, `tests/test_factory.py`, every real-weight test.

---

### DEC-013 — Import `gpt_oss_d_p`'s `utils/` helpers rather than copying them
- **Phase / module:** P2 / survey rows 3 and 4
- **Date (UTC):** 2026-09-04
- **Trigger:** the P3 tree says `utils/general_utils.py  # get_cache_file_name, get_default_num_links
  (copy from gpt_oss_d_p/utils)` and `utils/substate.py  # substate() state-dict prefix splitter`
  (`BRINGUP_RECIPE.md:806-808`) — but agent-contract rule 4 says reuse means *import*, not
  copy-paste, and a copy-paste needs a `DEC`.
- **Question:** copy the three helpers into `models/demos/llama31_8b_d_p/utils/`, or import them from
  `models/demos/gpt_oss_d_p/utils/`?
- **Options considered:**
  1. Copy, as the tree comment says. Removes a cross-package dependency; duplicates 110 lines that
     contain a real arch branch (`is_blackhole()` → 2 links, single-row mesh → 1) which would then
     have to be kept in sync by hand.
  2. Import. Zero duplication; couples this package to `models/demos/gpt_oss_d_p`, which is a peer
     demo, not a library.
  3. Import from a neutral home. There isn't one — `models/demos/common/` has the prefill engine but
     no such helpers.
- **Choice:** option 2 for `substate`, `get_cache_file_name`, `cache_file_exists` and
  `get_default_num_links`; `models/demos/llama31_8b_d_p/utils/` is created in P5 only if something
  genuinely Llama-specific needs to live there.
- **Why:** rule 4 is explicit and these four functions have zero model-specific content. The coupling
  is real but small and visible; `models/demos/minimax_m3` and `models/demos/gpt_oss_d_p` already
  cross-import the DeepSeek substrate the same way (`models/demos/gpt_oss_d_p/tt/mlp.py:21`), so it
  is the tree's existing convention rather than a new one.
- **Evidence:** `models/demos/gpt_oss_d_p/utils/general_utils.py:11`, `:15`, `:27` (35 lines total,
  no gpt-oss dims anywhere); `models/demos/gpt_oss_d_p/utils/substate.py:15`;
  `models/demos/gpt_oss_d_p/tt/mlp.py:21` (the precedent).
- **Confidence:** medium — this is a deliberate deviation from a tree comment, and the counter-case
  (a peer demo is not a stable dependency) is legitimate.
- **Falsifier:** `models/demos/gpt_oss_d_p` changing `get_default_num_links`' semantics for a
  gpt-oss-specific reason and silently changing this model's link count. Cheap detection:
  `G-SEMAPHORE` and `G-FABRIC-MATRIX` both pin the link count explicitly.
- **Revisit if:** the P7 unification lands (`models/demos/gpt_oss_d_p/README.md:26` tracks hoisting
  this scaffolding into `models/demos/common/prefill`), which would give these helpers a proper home.
- **Blast radius:** every module that loads a weight or calls a collective; `G-MESH`, `G-WEIGHTS`.

---

### DEC-014 — `tt/rms_norm.py` is an adaptation, and it adds an explicit `compute_kernel_config`
- **Phase / module:** P2 / survey row 5 (executed in P5.2)
- **Date (UTC):** 2026-09-04
- **Trigger:** `models/demos/gpt_oss_d_p/tt/rms_norm.py` is a structural match for Llama's plain
  RMSNorm, but calling `ttnn.rms_norm` the way it does would inherit a measured precision loss.
- **Question:** import the gpt-oss `RMSNorm` as-is, or write an adapted one?
- **Options considered:**
  1. Import it. It already has the `use_gemma_norm=False` branch Llama needs. But it reads
     `hf_config.rms_norm_eps` as an *attribute* (`models/demos/gpt_oss_d_p/tt/rms_norm.py:46`) —
     incompatible with this package's dict `hf_config` (P1 trap 2) — and it passes **no**
     `compute_kernel_config` at `:94`.
  2. Adapt: same structure, dict config, explicit `compute_kernel_config` with
     `fp32_dest_acc_en=True`, Gemma fold deleted.
- **Choice:** option 2.
- **Why:** §2.4 of the recipe measures `ttnn.rms_norm` with no config at **0.9999652** and with
  HiFi4 + `fp32_dest_acc_en=True` at **0.9999971** against a floor of 0.9999986 — the flag removes
  ~25x of the error, and at module level it moves `G-RMS` from 0.9999697 to 0.9999955. An imported
  module that cannot be given the config is an imported precision regression. The dead Gemma branch
  would also be exactly the kind of copied-in feature `00_MODEL_CARD.md` §3 exists to stop.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/rms_norm.py:33` (`is_distributed` pinned False with the
  condition commented out), `:46` (attribute access), `:94` (`ttnn.rms_norm` with no
  `compute_kernel_config`); recipe §2.4's A/B table.
- **Confidence:** high.
- **Falsifier:** `G-RMS` measuring no difference between the two configurations on this box — the
  A/B is run in-suite in P5.2 precisely so the claim is a number and not a quotation.
- **Revisit if:** the distributed (3-op) norm branch is ever switched on (residual scheme B, P4).
- **Blast radius:** `tt/rms_norm.py`, `G-RMS`, and by precedent every other module's config.

---

### DEC-015 — RoPE: import the llama3 scaling math, adapt only the table builder
- **Phase / module:** P2 / survey row 6 (executed in P5.3)
- **Date (UTC):** 2026-09-04
- **Trigger:** two candidate sources — `models/tt_transformers/tt/common.py`'s llama3-aware
  `precompute_freqs`/`apply_scaling`, and `models/demos/gpt_oss_d_p/tt/rope.py`'s builder, which is
  YaRN end to end.
- **Question:** which parts are imported and which are written?
- **Options considered:**
  1. Adapt gpt-oss's `rope.py` wholesale, replacing YaRN with llama3. Would re-implement scaling
     math that already exists and is llama-specific *in the llama package*.
  2. Import `precompute_freqs(..., rope_type="llama3")` for the frequencies and write a thin
     table/transformation-matrix adapter around it.
- **Choice:** option 2.
- **Why:** the scaling math is the part that is easy to get subtly wrong (the piecewise band), and
  `G-REF` measured the repo helper **bit-identical** (`max|Δ| = 0.0`) to this package's independent
  transcription of `_compute_llama3_parameters`, so it is verified, not merely available. What
  remains — cos/sin table layout, Meta interleave, the transformation matrix — is thin.
- **Evidence:** `models/tt_transformers/tt/common.py:489`, `:437`; `G-REF` raw log
  `raw/G-REF_20260904T035140Z.log` ("llama3 inv_freq vs tt_transformers precompute_freqs: max|delta|
  0.000e+00"); `models/demos/gpt_oss_d_p/tt/rope.py:36` (`yarn_inv_freq`, not reusable).
- **Confidence:** high.
- **Falsifier:** `G-ROPE` failing past `original_max_position_embeddings` while the reference alone
  agrees with HF — which would point at the repo helper's hard-coded factors (`R-010`).
- **Revisit if:** a second llama3-scaled model with different low/high frequency factors is added
  (see `R-010`).
- **Blast radius:** `tt/rope.py`, `G-ROPE`, `G-ATTN`, `G-MODEL` at long context.

---

### DEC-016 — Three separate Q/K/V projections, not a fused QKV
- **Phase / module:** P2 / survey row 10 (executed in P5.5)
- **Date (UTC):** 2026-09-04
- **Trigger:** the source patterns disagree — `models/demos/gpt_oss_d_p/tt/attention/weights.py:23`
  keeps three column-parallel weights, while `models/tt_transformers/tt/load_checkpoints.py:494`
  ships `fuse_qkv_meta` for a single fused weight.
- **Question:** load `q_proj`/`k_proj`/`v_proj` as three weights, or pre-fuse into one
  `[hidden, (nq + 2*nkv)*head_dim]` weight and split heads on device?
- **Options considered:**
  1. Three separate matmuls — simplest, matches the attention template, 3 matmul launches per layer.
  2. Fused QKV — one matmul, needs `ttnn.experimental.nlp_create_qkv_heads` to split a GQA 32/8
     layout, and a second thing to debug before any PCC exists.
- **Choice:** option 1.
- **Why:** this iteration is functional-first (recipe non-goals: no perf work). The fused path adds a
  head-split whose GQA layout is an independent failure mode, and it changes the weight-cache layout,
  so switching later is a cache invalidation, not a rewrite.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/attention/weights.py:38` `load_attention_weights` (the
  three-weight pattern, PCC-verified by `models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py`);
  `models/tt_transformers/tt/load_checkpoints.py:494` `fuse_qkv_meta` (the alternative, available).
- **Confidence:** high.
- **Falsifier:** `G-ATTN` passing while per-layer wall clock is dominated by matmul launch overhead —
  which would make this the wrong choice for the *next* iteration, not for this one.
- **Revisit if:** perf work starts, or the SP ring-SDPA path requires a fused Q/K/V layout.
- **Blast radius:** `tt/attention/weights.py`, `tt/attention/prefill.py`, the weight cache, `G-ATTN`.

---

### DEC-017 — The P3 tree wins: no `docs/`, no `scripts/__init__.py`, no vendored recipe
- **Phase / module:** P3 / package skeleton
- **Date (UTC):** 2026-09-04
- **Trigger:** `DEC-002` deferred the recipe's self-conflict to P3, and `06_GATES.md`'s P2 status line
  names it as the first thing P3 must settle. P0 step 1 (`BRINGUP_RECIPE.md:485`) says create
  `{tt,tests/unit,scripts,docs}` *with `__init__.py` files*; the P3 target tree
  (`BRINGUP_RECIPE.md:752-819`) has no `docs/` and no `scripts/__init__.py`, and lists a package-local
  `BRINGUP_RECIPE.md` (`:754`) that `DEC-002` declined to create.
- **Question:** which spelling does the committed tree use, and when do the two P0 artefacts go?
- **Options considered:**
  1. Keep both and let `G-CLEAN` (P9) decide. Defers a two-file question past six phases and leaves
     two files that every intermediate phase's "no dead files" reading has to re-litigate.
  2. Commit to the P3 tree now, in the phase that *owns* the tree, and delete both in **P5.1** — the
     first phase that touches the tree at all. P3 produces a document; deleting files inside it would
     make a doc-review gate carry a code change.
  3. Delete them inside P3. Smallest total diff, but it makes `G-OUTLINE` (a doc gate) the gate for a
     file deletion, and P3's raw log would then have to prove a tree change it is not scoped to make.
- **Choice:** option 2. The committed tree is `03_OUTLINE.md` §1; `docs/` (with its `.gitkeep`) and
  `scripts/__init__.py` are **deleted in P5.1**, and `G-CLEAN` items 1 and 5 re-check. The recipe is
  **not** vendored (re-affirming `DEC-002`), recorded as `[DEV-1]`.
- **Why:** P0's step and P3's tree are two statements of the same requirement at different
  specificity, and the tree is the one that describes the *end state* — which is what a later reviewer
  and `G-CLEAN` compare against. Neither template ships either file
  (`models/demos/gpt_oss_d_p/scripts/`, `models/demos/minimax_m3/scripts/` have no `__init__.py`;
  neither has a `docs/`). `scripts/verify_citations.py` is *run*, never imported, so removing
  `scripts/__init__.py` cannot break a call site.
- **Evidence:** `BRINGUP_RECIPE.md:485` vs `BRINGUP_RECIPE.md:752-819`;
  `models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py` and
  `models/demos/minimax_m3/scripts/verify_golden_kv.py:26` both live in `__init__.py`-free `scripts/`
  directories; `03_OUTLINE.md` §1.1 `[DEV-1]`/`[DEV-3]`.
- **Confidence:** high — `DEC-002` rated the same question *medium* because it was a coin flip between
  two recipe statements; P3 owns the tree, so it is no longer a coin flip.
- **Falsifier:** a P5–P10 file needing `scripts/` to be an importable package (e.g. a test importing
  `generate_golden_kv_cache`), or a genuine need for package documentation outside `README.md` and
  `bringup_log/`.
- **Revisit if:** P5.1 runs (it performs the deletion), or P9's `G-CLEAN` finds either file still present.
- **Blast radius:** two files; `G-OUTLINE`, `G-CLEAN`. Supersedes the deferral in `DEC-002`, not its
  reasoning.

---

### DEC-018 — No `utils/` package: the four helpers stay imported
- **Phase / module:** P3 / package skeleton
- **Date (UTC):** 2026-09-04
- **Trigger:** `DEC-013` (P2) chose to **import** `substate`, `get_cache_file_name`,
  `cache_file_exists` and `get_default_num_links` from `models/demos/gpt_oss_d_p/utils/` rather than
  copy them, against the recipe tree's `# (copy from gpt_oss_d_p/utils)` comment
  (`BRINGUP_RECIPE.md:786-788`). P3 must say whether the directory exists at all.
- **Question:** does the committed tree contain `utils/` (with an `__init__.py` and nothing else), or
  no `utils/` at all?
- **Options considered:**
  1. Create `utils/__init__.py` now as a placeholder, so the tree matches the recipe's shape and a
     future Llama-specific helper has a home. An empty package is exactly the dead file `G-CLEAN`
     item 5 rejects, and it would mislead a reader into looking there for `substate`.
  2. No `utils/` at all. `03_OUTLINE.md` §1.1 `[DEV-2]` records the absence and names the import
     source, so the absence reads as a decision rather than an omission. If P5 finds something
     genuinely Llama-specific, `utils/` is created **then**, in the edit that has the first occupant.
- **Choice:** option 2.
- **Why:** an empty package communicates nothing that the outline's deviation table does not
  communicate better, and it costs a `G-CLEAN` exemption. The four functions have zero
  model-specific content (`models/demos/gpt_oss_d_p/utils/general_utils.py` is 35 lines and mentions
  no gpt-oss dimension; `models/demos/gpt_oss_d_p/utils/substate.py:15` is pure dict manipulation).
- **Evidence:** `models/demos/gpt_oss_d_p/utils/general_utils.py:11`, `:15`, `:27`;
  `models/demos/gpt_oss_d_p/utils/substate.py:15`; the cross-package-import precedent
  `models/demos/gpt_oss_d_p/tt/mlp.py:21`; `DEC-013`.
- **Confidence:** high for the *directory*; the underlying import-vs-copy call keeps `DEC-013`'s
  *medium* (a peer demo is not a stable dependency).
- **Falsifier:** two or more Llama-specific helpers appearing in P5–P7 with nowhere natural to live,
  which would mean the directory should have existed from the start.
- **Revisit if:** `DEC-013` is reversed, or `models/demos/gpt_oss_d_p/README.md:26`'s P7 unification
  lands and gives these helpers a neutral home.
- **Blast radius:** the tree; every module's import block; `G-CLEAN` item 5.

---

### DEC-019 — GQA head split with three separate projections — and a correction to `DEC-016`'s evidence
- **Phase / module:** P3 / `tt/attention/{weights,operations}.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** P3 has to write the head-split signature, and the two P2 rows that describe it are
  mutually inconsistent. `02_SURVEY.md` row 9 says
  `models/demos/gpt_oss_d_p/tt/attention/weights.py:23` keeps "the same three-weight column-parallel"
  shape; row 11 says to import `split_qkv_heads_prefill`
  (`models/demos/gpt_oss_d_p/tt/attention/operations.py:29`), which wraps
  `ttnn.experimental.nlp_create_qkv_heads` — an op that consumes a **fused** QKV tensor.
- **What is actually true, and it contradicts both the survey and the recipe's own worked example:**
  `models/demos/gpt_oss_d_p/tt/attention/weights.py:83-100` builds a **single fused** per-device
  `wqkv` (chunk q/k/v across TP, transpose, `torch.cat` per device, concat across devices) and
  `models/demos/gpt_oss_d_p/tt/attention/operations.py:25` runs **one** `ttnn.linear` on it. So
  gpt-oss is the *fused* pattern, not the three-weight pattern. `BRINGUP_RECIPE.md:213-222` — the
  recipe's own template `DEC` block — cites `models/demos/gpt_oss_d_p/tt/attention/weights.py` as
  evidence for the **three-weight** option and names `nlp_create_qkv_heads` as something only the
  *fused* option needs; both halves are backwards. `DEC-016` inherited that reading.
- **Question:** keep `DEC-016`'s three separate projections, and if so, what performs the head split?
- **Options considered:**
  1. **Three separate `q/k/v_proj` weights + `nlp_create_qkv_heads(q, concat(k, v))`.** The op's
     two-tensor branch (`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp:22-32`)
     takes Q on its own and a **K|V** second tensor, so three projections *are* compatible with it at
     the cost of one on-device `ttnn.concat` on dim 3. 3 matmuls + 1 concat + 1 head-split per layer.
  2. Three separate weights + a hand-rolled `reshape` + `permute` per tensor. No concat, but three
     transposes and a layout the op is known-good at, re-derived by hand.
  3. Fused per-device `wqkv` at load time (gpt-oss's real pattern) + one `nlp_create_qkv_heads`.
     1 matmul + 1 head-split; the fewest device ops. Needs the per-device Q|K|V interleave to be
     exactly right, and it changes the weight-cache layout.
- **Choice:** option 1. `DEC-016`'s choice **stands**; its evidence is corrected here.
- **Why:** `DEC-016`'s actual reason — functional-first, and don't debug a GQA layout permutation
  before any PCC exists — survives the correction intact, and option 1 keeps the loader trivially
  auditable (`G-WEIGHTS` asserts bit-exactness *through* the transpose and the Meta swizzle; a
  per-device interleave is a third transformation to prove). Option 3 is the perf follow-up and it is
  a **cache invalidation**, not a rewrite. Option 2 re-derives on device what a maintained op already
  does.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/attention/weights.py:96` (`torch.cat([wq, wk, wv], dim=-1)`
  — the fusion), `models/demos/gpt_oss_d_p/tt/attention/operations.py:25` (one `ttnn.linear` on
  `weights.wqkv`), `models/demos/gpt_oss_d_p/tt/attention/operations.py:41`
  (`nlp_create_qkv_heads` on that fused tensor);
  `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp:26-32`
  (the two-tensor Q + K|V branch and its head-dim equality check);
  `models/tt_transformers/tt/load_checkpoints.py:494` `fuse_qkv_meta` (the fused alternative, still
  available).
- **Confidence:** high on the correction (read from the source, twice); medium on the choice, because
  the fused path is what both the recipe's non-example prose and the nearest template actually do, and
  the concat is a real extra op.
- **Falsifier:** `G-ATTN` showing the `ttnn.concat` dominating the block's wall clock, or
  `nlp_create_qkv_heads` refusing the two-tensor form at `head_dim = 128` / `nq_local = 4` /
  `nkv_local = 1` (it would refuse loudly — the check is a `TT_FATAL`).
- **Revisit if:** perf work starts, or `G-ATTN` needs the fused layout for the SP ring path.
- **Blast radius:** `tt/attention/weights.py`, `tt/attention/operations.py`, the weight cache layout,
  `G-ATTN`, `G-WEIGHTS`. **Corrects the *Evidence* field of `DEC-016`; does not supersede its choice.**

---

### DEC-020 — `head_dim` is derived in the one normalised constructor, not read from the config
- **Phase / module:** P3 / `tt/model_config.py` (executed in P6.2, consumed from P5.5)
- **Date (UTC):** 2026-09-04
- **Trigger:** the nearest template reads `hf_config.head_dim`
  (`models/demos/gpt_oss_d_p/tt/model.py:64`) and Llama-3.1-8B's `config.json` has **no `head_dim`
  key** (`00_MODEL_CARD.md` §2 records it as derived). P1 trap 2 already required one normalising
  constructor; P3 has to say what it produces.
- **Question:** where does `head_dim = 128` come from, and what happens if a future checkpoint carries
  an explicit `head_dim` that disagrees with the derivation?
- **Options considered:**
  1. `getattr(hf_config, "head_dim", hidden // heads)`. Exactly the shape of the `rope_theta` trap
     (`07_RISKS.md` R-005): a missing key silently becomes a computed default, and a *wrong* key
     silently wins.
  2. Derive unconditionally in `ModelArgs.__init__` (`hidden_size // num_attention_heads`) and
     **assert** that any explicit `head_dim` in the config equals the derivation.
  3. Hard-code 128. Fails the provenance rule (agent-contract rule 3) and breaks on any other Llama size.
- **Choice:** option 2. `4096 // 32 = 128`, asserted against the config when the key exists, exposed
  once as `ModelArgs.head_dim`, and every module (`AttentionConfig`, `kv_cache`, `rope`) takes it from
  there.
- **Why:** the derivation is verifiable arithmetic and the assert converts the one dangerous case (a
  checkpoint where the two disagree — e.g. Llama-3.2's `head_dim` key) from silent wrongness into a
  build-time failure. Option 1 is the trap the whole of P1 §4 is about, one attribute over.
- **Evidence:** `00_MODEL_CARD.md` §2 (`head_dim | 128 | derived: hidden_size / num_attention_heads
  = 4096 / 32 = 128. No head_dim key exists in this config.json`);
  `models/demos/gpt_oss_d_p/tt/model.py:64` (the attribute read that would raise here);
  `models/tt_transformers/tt/common.py:165` (the same dict-helper discipline for theta).
- **Confidence:** high.
- **Falsifier:** a Llama checkpoint whose explicit `head_dim` legitimately differs from
  `hidden/heads` — the assert would then be wrong rather than protective. (Llama-3.2-1B/3B are such
  models by key presence, not by value; the assert passes on both.)
- **Revisit if:** the package is pointed at a non-8B Llama, or `ModelArgs` lands in P6.2.
- **Blast radius:** `tt/model_config.py`, `tt/attention/config.py`, `tt/attention/kv_cache.py`,
  `tt/rope.py`; every shape in `03_OUTLINE.md` §3.

---

### DEC-021 — KV cache dtype is `bfloat8_b`, with the bf16 delta owed at `G-KV`
- **Phase / module:** P3 / `tt/attention/kv_cache.py` (measured in P5.6)
- **Date (UTC):** 2026-09-04
- **Trigger:** `03_OUTLINE.md` §3's shape table cannot be written without a cache dtype, and
  `BRINGUP_RECIPE.md:1193-1197` says "`bfloat8_b` is the cache dtype (P3, and every threshold in
  Appendix A assumes it); bf16 is a measurement mode... Log that delta as a `DEC` — the PCC cost
  measured, not assumed".
- **Question:** which dtype does the cache hold, and how is the choice justified before P5.6 measures
  anything?
- **Options considered:**
  1. `bfloat8_b`. Halves the cache footprint (`[users*32, 1, max_seq_len/4, 128]` per chip is the
     largest allocation in the model), matches the DeepSeek substrate the write op comes from
     (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:56` defaults to it) and is what every
     Appendix A KV threshold was set against (`G-KV` ≥ 0.99, `G-CHUNK`/`G-KV-TP8` K ≥ 0.99 / V ≥ 0.98).
  2. `bfloat16`. Higher fidelity, 2x the DRAM, and **every KV threshold in Appendix A would have to be
     re-derived** — which the recipe forbids doing after seeing a measurement (§A.2).
- **Choice:** option 1, and the **bf16 A/B is run as a measurement mode at `G-KV`** rather than
  skipped: `G-KV` records the bf8_b number, the bf16 number, and the error ratio of each to its own
  computed floor.
- **Why:** the thresholds this bring-up is gated on already assume bf8_b, so choosing bf16 here would
  mean either re-fitting thresholds (forbidden) or gating a bf16 cache on bf8_b numbers (meaningless).
  The choice is therefore made by the gate design, and what P3 owes is the *measurement plan*, not a
  guess at the cost.
- **Evidence:** `BRINGUP_RECIPE.md:1193-1197`; Appendix A rows `G-KV` / `G-KV-TP8` / `G-MESH-KV`
  (`BRINGUP_RECIPE.md:1755`, `:1740`, `:1746`); `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:56`
  (`cache_dtype=ttnn.bfloat8_b` default) and `:72` (its comment: "bf8 matches the DeepSeek substrate
  + the device golden check").
- **Confidence:** high on the choice, **none yet on the cost** — the delta is unmeasured, which is
  exactly why it is named here as a P5.6 obligation rather than asserted.
- **Falsifier:** `G-KV`'s bf8_b run sitting far off its own computed floor while the bf16 run sits on
  it — that would mean the loss is not storage rounding but something in the write path, and lowering
  the threshold would be the wrong response.
- **Revisit if:** `G-KV` or `G-CHUNK` misses its threshold at bf8_b (the recipe's instruction is then
  to keep the threshold and log a `DEC`, not to lower it), or long-context DRAM pressure changes the
  footprint argument.
- **Blast radius:** every KV threshold in Appendix A; `tt/attention/kv_cache.py`; `G-KV`, `G-CHUNK`,
  `G-KV-TP8`, `G-MESH-KV`, `G-MOCK-MIG`.

---

### DEC-022 — Activation dtype ladder: bf16 residual stream, bf8_b weights
- **Phase / module:** P3 / every module
- **Date (UTC):** 2026-09-04
- **Trigger:** `03_OUTLINE.md` §3 has a dtype column, and the templates set activation dtype in three
  different places with two different rules.
- **Question:** what dtype does the residual stream carry, what do the weights carry, and where does
  bf8_b appear?
- **Options considered:**
  1. bf16 activations throughout, bf8_b weights and KV cache. What both templates do for the residual
     (`models/demos/gpt_oss_d_p/tt/model.py:315` embeds at bf16 with the comment that bf8's per-tile
     shared exponent "crushes small channels once massive activations appear").
  2. bf8_b activations as well, for footprint. gpt-oss switches to bf8_b activations **only** past
     `seq_len > 32*1024` (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:106-109`).
  3. bf16 weights too. Halves the achievable PCC advantage away for 2x the DRAM, and Appendix A's
     `G-MLP` explicitly gates **both** dtypes, so the weights must be bf8_b-capable regardless.
- **Choice:** option 1, with gpt-oss's long-sequence escape hatch **not** carried into this iteration:
  activations bf16 everywhere, weights bf8_b (both dtypes exercised at `G-MLP`), KV cache bf8_b
  (`DEC-021`), norm weights bf16 `ROW_MAJOR`, logits bf8_b.
- **Why:** the residual stream is the one tensor every layer accumulates into, so it is where a
  shared-exponent format does the most damage, and it is the cheapest tensor to keep wide. Dropping
  the `seq_len > 32*1024` bf8_b switch keeps the dtype of a gate's output independent of its sequence
  length — otherwise `G-MODEL` at long context would silently be measuring a different numeric path
  from `G-MODEL` at 512.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/model.py:313-315` (the bf16-not-bf8 comment on the
  embedding output); `models/demos/gpt_oss_d_p/tt/attention/prefill.py:106-109` (the long-sequence
  switch, not taken); Appendix A `G-MLP` "≥ 0.999 @bf8_b, ≥ 0.9995 @bf16" (`BRINGUP_RECIPE.md:1753`).
- **Confidence:** high.
- **Falsifier:** a long-context (>32k) prefill OOMing where the bf8_b activation switch would have
  fitted — which would make this a footprint decision wrongly made on numerics.
- **Revisit if:** long-context work starts, or `G-MLP`'s bf8_b weight number misses its threshold
  (which the recipe says to answer with a `DEC` keeping bf16 weights, not a lowered threshold).
- **Blast radius:** `03_OUTLINE.md` §3; every module's `dtype=` argument; `G-MLP`, `G-ATTN`,
  `G-LAYER`, `G-MODEL`.

---

### DEC-023 — One bring-up env var, `LLAMA_DELTA_PROBE`
- **Phase / module:** P3 / `tt/layer.py` (executed in P6.1)
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:1249-1250` requires the per-layer residual-delta probe behind "one
  env var" and its output in `bringup_log/raw/`; introducing an env var is a mandatory `DEC` (§1.3)
  and must appear in the README's table.
- **Question:** what is the variable called, and does the package add any others?
- **Options considered:**
  1. `LLAMA_DELTA_PROBE`, mirroring `GPT_OSS_DELTA_PROBE`
     (`models/demos/gpt_oss_d_p/tt/layer.py:19`). Consistent with the template's naming; obviously
     package-scoped.
  2. A pytest `--delta-probe` flag instead. Would not work for the P8 galaxy harnesses and the P10
     two-terminal runs, which are not pytest.
  3. Reuse a `PREFILL_*` name. Those belong to the engine, and `tt-run` forwards only
     `TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_` prefixes anyway (`BRINGUP_RECIPE.md:1831`), so a
     `PREFILL_`-prefixed package variable would be silently dropped under `tt-run`.
- **Choice:** option 1. The package's **total** env-var surface is planned as four —
  `HF_MODEL` (pre-existing, repo-wide), `TT_CACHE_PATH` (pre-existing, repo-wide),
  `LLAMA_DELTA_PROBE` (new here), and `PREFILL_TOPOLOGY` (P8, the name
  `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:121` already uses) — and P9 item 6
  regenerates that list by grep rather than trusting this count.
- **Why:** the probe has to work from pytest, from a bare python harness and from a two-terminal
  engine run, which only an environment variable does. Naming it after the package keeps it
  greppable and keeps it out of the engine's namespace.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/layer.py:19` (`_DELTA_PROBE = os.environ.get("GPT_OSS_DELTA_PROBE", "") != ""`)
  and `:22` (`_delta_stats`, whose `except Exception` is the one allowed instance under
  `BRINGUP_RECIPE.md:1708-1710` because it logs and must never break a run);
  `BRINGUP_RECIPE.md:1249-1250`.
- **Confidence:** high.
- **Falsifier:** P9 item 6's grep finding a fifth `os.environ` read in the package that this decision
  did not anticipate.
- **Revisit if:** P8 or P10 needs a knob that is not already an engine variable.
- **Blast radius:** `tt/layer.py`, `README.md`'s env-var table, `G-CLEAN` item 6.

---

### DEC-024 — Replicate the embedding table; do not TP-shard the vocab
- **Phase / module:** P4 / `tt/embedding.py` (executed in P6.2)
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:940` makes this an explicit `DEC`: "`Embedding` | after lookup |
  `all_gather` if the vocab is TP-sharded | `DEC` — or replicate the embedding table and skip it".
  P4 must decide it because the answer adds or removes a collective.
- **Question:** is `embed_tokens.weight` `[128256, 4096]` replicated on every chip, or sharded — and
  if sharded, on the vocab (rows) or the embedding dim (columns)?
- **Options considered:**
  1. **Replicated.** `128256 * 4096 * 2 B = 1.05 GiB` per chip, bf16, in DRAM. Zero collectives; the
     lookup yields the full 4096-wide row directly, which is exactly what scheme A's residual wants.
     What the nearest template does, with a standing TODO to shard it
     (`models/demos/gpt_oss_d_p/tt/model.py:82-83`).
  2. **Vocab-sharded (rows) across TP.** `16032` rows per chip; every chip's lookup misses for
     7/8 of the tokens, so it needs a masked lookup **plus** an all-reduce or all-gather to
     reassemble. Saves ~920 MiB per chip.
  3. **Emb-dim-sharded (columns) across TP.** Each chip holds `[128256, 512]` and produces its own
     512-wide slice — the natural input to scheme **B**, and what
     `models/demos/minimax_m3/tt/parallel_embedding.py:80` implements (with an optional second
     vocab-on-SP split this model does not need). Under scheme A it then needs a closing all-gather.
- **Choice:** option 1, replicated.
- **Why:** it is the only option with **no collective at all**, and under scheme A (`DEC-025`) the
  other two both pay one to get back to the full-width residual the norm and Q/K/V projections want.
  1.05 GiB per chip is affordable next to the KV cache, which is the allocation that actually scales
  with context. Option 3 becomes the right answer the day scheme B is taken — it produces `emb/tp`
  natively — which is why `DEC-025`'s blast radius includes this file.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/model.py:82-83` (the replicated table plus its "shard it
  later, deferred" TODO) and `:84` (the `as_tensor` with **no** `mesh_mapper`, i.e. replication);
  `models/demos/minimax_m3/tt/parallel_embedding.py:80` (the sharded alternative);
  `00_MODEL_CARD.md` §4.2 (`128256/8 = 16032 = 501*32`, so either split is tile-legal — the choice is
  not forced by alignment).
- **Confidence:** high for this iteration; the footprint argument is the part that dates.
- **Falsifier:** a DRAM OOM at long context traceable to the embedding table rather than the KV
  cache. That would make this a footprint decision wrongly made on op count.
- **Revisit if:** residual scheme B is taken (option 3 becomes free), or `num_users > 1` /
  long-context work makes DRAM the binding constraint.
- **Blast radius:** `tt/embedding.py`, `tt/model.py`, `03_OUTLINE.md` §3's embedding row,
  `04_CCL_PLAN.md` §4 (one row becomes "none"), `G-WEIGHTS`.

---

### DEC-025 — Residual layout: scheme A (replicated full-emb), on cost equivalence
- **Phase / module:** P4 / every module
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:947-949` requires the choice to be made consciously in P4 and held
  everywhere, because it touches every module's output layout and every norm's input layout.
- **Question:** does the residual stream carry full `[1,1,S_loc,4096]` on every TP column (**A**), or
  `[1,1,S_loc,512]` = `emb/TP` (**B**)?
- **Options considered:**
  1. **A — replicated.** Attention and MLP close with a full **all-reduce** (RS + AG); norms are
     single-op and local; embedding is replicated.
  2. **B — sharded.** Attention and MLP close with a **reduce-scatter only**; each norm either
     all-gathers its input first (`DEFAULT_NORM_MODE = "gather_first"`,
     `models/demos/minimax_m3/tt/residual.py:32`) or runs the 3-op distributed RMSNorm on the shard.
     Requires `4096/8 % 32 == 0`, satisfied (`512 = 16*32`).
- **Choice:** **A**, for this iteration, with B's seam (`scatter_output`) wired from day one.
- **Why — and the obvious reason is the wrong one.**
  - **Wrong reason:** "B is unproven, because `models/demos/gpt_oss_d_p/tt/rms_norm.py:33` pins
    `is_distributed = False` with the condition commented out." That branch is dormant, but B does
    not need it: `models/demos/minimax_m3/tt/residual.py:26` ships **B on by default** with
    `gather_first`, and its own comment says B measured better there on device time and op launches
    with the KV PCC bit-identical. Only **B-with-distributed-norm** is unproven.
  - **Right reason:** on a *dense* model the two schemes issue the **identical** collectives per
    layer — 2 reduce-scatters + 2 all-gathers, same sizes, same axis. B's advantage in Minimax comes
    from sharing one gathered norm output across **several MoE consumers**
    (`models/demos/minimax_m3/tt/residual.py:9-11`); Llama has no such consumers — each norm output
    feeds exactly one module. So B buys nothing here and costs a second layout to reason about in
    every gate.
  - Two secondary reasons: A keeps `G-TP-PARITY` a direct device-vs-device comparison at full width
    (at `emb/tp` the single-card output would need re-slicing before comparison), and a replicated
    embedding (`DEC-024`) already hands the first layer a full-width residual.
- **Evidence:** `models/demos/minimax_m3/tt/residual.py:9-11` (the shared-gather argument that does
  not apply here), `:26` (B is the default there), `:32` (`gather_first`);
  `models/demos/gpt_oss_d_p/tt/rms_norm.py:33` (the dormant branch — the *wrong* argument);
  `models/demos/minimax_m3/tt/dense_mlp.py:38` (`scatter_output=None`) and `:96-112` (the two-branch
  tail this package copies); `04_CCL_PLAN.md` §6.
- **Confidence:** high on the equivalence argument; medium that A stays right past this iteration —
  B halves the activation footprint of the residual stream, which matters at long context even
  without a shared-gather win.
- **Falsifier:** a measured per-layer device-time or op-launch difference between the two schemes on
  a *dense* layer. The cost-equivalence claim is an op-count argument, not a measurement, and it is
  falsifiable by running the `scatter_output` seam at P8.
- **Revisit if:** long-context DRAM pressure bites, the distributed-norm branch is validated, or a
  second consumer appears downstream of any norm.
- **Blast radius:** every module's output layout; `tt/rms_norm.py`'s `is_distributed` default;
  `tt/embedding.py` (`DEC-024`); `03_OUTLINE.md` §3's residual row; `04_CCL_PLAN.md` §5 rows 3, 4, 6;
  `G-TP-PARITY`, `G-LAYER`, `G-MODEL`.

---

### DEC-026 — Ship the barrier ping-pong at depth 2, and do not reset it between chunks
- **Phase / module:** P4 / `tt/ccl.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:905-909` requires a `DEC` "either way" on two coupled facts about
  the template's barrier semaphores: the ping-pong is only **2 deep**, and
  `reset_global_semaphores` deliberately **skips** the barrier and ring-attention semaphores — while
  chunked prefill **does** reuse one `CCLManager` across chunks.
- **Question:** deepen the barrier ring to 4 and/or reset it per chunk now, or ship the template's
  behaviour and let `G-RACE` decide?
- **Options considered:**
  1. **Ship depth 2, no reset** — the template verbatim
     (`models/demos/gpt_oss_d_p/tt/ccl.py:77`, `:132`). Known-good on this exact box at this exact
     mesh for a 36-layer chunked prefill, which is more evidence than any change would have.
  2. **Deepen to 4 pre-emptively.** 4 barrier semaphores instead of 2; wider gap between reuses.
     But it is a change made *before* the measurement that would justify it, and if `G-RACE` then
     passes, nobody learns whether depth 2 was ever the problem — the recipe's own rule against
     picking a number before its measurement, applied to a resource count.
  3. **Reset the barrier semaphores per chunk.** Removes the "stale state across chunks" worry, adds
     a host-side call on the per-chunk path, and diverges from the substrate the write op comes from.
- **Choice:** option 1. Depth stays **2**; `reset_global_semaphores` keeps skipping the barrier and
  ring-attention sets. **`G-RACE` is the measurement**, and if it fails, deepening 2 → 4 is the
  documented **first** move — before suspecting the model (`BRINGUP_RECIPE.md:908-909`).
- **Why:** the arithmetic is thin but not obviously wrong — RS takes `barrier[0]`, the following AG
  takes `barrier[1]`, the next RS takes `barrier[0]` again, so every reuse has exactly one op of
  separation, 64 times per 32-layer forward under scheme A. The upstream TODO's reasoning ("one-shot
  prefill never reuses a `CCLManager` across runs") is *false* for chunked prefill, which is precisely
  why this is logged rather than assumed safe: the gate that would catch it exists (`G-RACE`, 3 runs
  in one process on one `CCLManager`, asserted **bit-identical**), and pre-emptively changing the
  resource would blind it.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/ccl.py:77` (`barrier_ns_sems = 2 * 1`), `:102-106`
  (the modulo-2 cycling), `:132` ("this deliberately does NOT reset the barrier or ring-attention
  semaphores ... TODO"); `models/demos/minimax_m3/config.py:102` and `:124` (both halves of an
  all-reduce take a barrier semaphore, which is where the 4-per-layer count comes from);
  `04_CCL_PLAN.md` §3.
- **Confidence:** medium. This is a deliberate bet that a known-good configuration is safer than an
  untested improvement, and the counter-case (a race that only shows up at 32 layers x N chunks, i.e.
  beyond `G-RACE`'s "hundreds of all-reduces, not hundreds of thousands" scope) is legitimate.
- **Falsifier:** `G-RACE`'s three runs not producing one hash, or any run-to-run multi-device PCC
  variation — Appendix B's first-listed cause for that symptom is exactly this.
- **Revisit if:** `G-RACE` fails; multi-user prefill lands (more chunks per `CCLManager`); or the
  upstream TODO at `models/demos/gpt_oss_d_p/tt/ccl.py:134` is closed.
- **Blast radius:** `tt/ccl.py`; every collective; `G-RACE`, `G-SEMAPHORE`, `G-MESH-KV`.

---

### DEC-027 — Fabric mode and topology are selected together, from one variable
- **Phase / module:** P4 / `tt/ccl.py`, `tests/{fabric_topology_matrix,galaxy_prefill_kv_pcc}.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** P4 must state a topology per collective call site, and a `Topology.Ring` collective on
  a plain `FABRIC_1D` fabric **hangs rather than erroring** (`BRINGUP_RECIPE.md:82-83`). A hang on
  this box is not contained — every later collective hangs too, until `tt-smi -r`.
- **Question:** what topology do the collectives use, how is the fabric configured to match, and which
  mesh-graph descriptor is pinned?
- **Options considered:**
  1. **Two independent knobs** (fabric config in the harness, `topology=` in the `CCLManager`
     constructor). Maximum flexibility and exactly the mismatch that hangs the box.
  2. **One variable selects both**, as the working template does: `PREFILL_TOPOLOGY=ring|linear`
     drives `ttnn.set_fabric_config(FABRIC_1D_RING | FABRIC_1D)`
     (`models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:121-122`) **and** the `Topology`
     handed to the manager (`:161`). Ring is the default.
  3. **Hard-code Ring.** Removes the mismatch, and removes the ability to bisect a Ring-specific
     hang against a Linear baseline — which is `G-FABRIC-MATRIX`'s whole job.
- **Choice:** option 2, reusing the template's variable name `PREFILL_TOPOLOGY` (default `ring`), so
  the package adds no new name for a knob that already exists. Default at `(4,8)`:
  `FabricConfig.FABRIC_1D_RING` + `ttnn.Topology.Ring` + `num_links = 2`.
  **The descriptor is deliberately not pinned yet**: `G-FABRIC-MATRIX` measures which of the three
  BH-galaxy torus descriptors works, and P8 pins the winner in the P10 manifest. The template's own
  harness uses
  `tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto`,
  which the recipe does not name (it names
  `tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sp4_torus_xy_graph_descriptor.textproto` and
  `tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto`);
  all three exist, so this is a measurement, not a reading.
- **Why:** coupling the two removes an entire failure class from the code rather than from a comment,
  and the failure class in question poisons the whole box. Keeping `linear` selectable is what makes
  `G-FABRIC-MATRIX` able to record "this case is expected to hang" as a *measurement*, in a
  subprocess with a timeout, instead of as a lost session.
- **Evidence:** `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:121`
  (`os.getenv("PREFILL_TOPOLOGY", "ring")`), `:122` (the paired `set_fabric_config`), `:161` (the
  paired `Topology`); `models/demos/gpt_oss_d_p/tests/test_kv_cache_table.py:126` (the
  `device_params` `fabric_config` parametrisation for pytest);
  `models/demos/gpt_oss_d_p/utils/general_utils.py:33` (single-row meshes get 1 link, so `(1,N)` runs
  never touch the deployment fabric); `BRINGUP_RECIPE.md:82-83`, `BRINGUP_RECIPE.md:1436-1442`.
- **Confidence:** high on the coupling; low on which descriptor — that is explicitly unmeasured and
  is `G-FABRIC-MATRIX`'s output.
- **Falsifier:** `G-FABRIC-MATRIX` finding a (mesh, topology, links) combination that works only with
  a *mismatched* pair, which would mean the coupling is too strict.
- **Revisit if:** `G-FABRIC-MATRIX` runs (it pins the descriptor), or P10's manifest needs the
  descriptor path — note a manifest cannot set `TT_MESH_GRAPH_DESC_PATH` for the runner's own process,
  so both the manifest and the binding's `global_env` may need it.
- **Blast radius:** `tt/ccl.py`'s `topology` argument; every P8 harness; `README.md`'s env-var table
  (4th entry); `G-FABRIC-MATRIX`, `G-TP-PARITY`, `G-MESH-KV`, `G-RACE`, `G-REQUEST`.
  **Also carries P8 step 2's `quiesce_devices()` requirement:** `G-TP-PARITY` compares `(1,1)`
  against `(1,TP)` and therefore holds two overlapping submeshes; `parent.quiesce_devices()`
  (`tt_metal/api/tt-metalium/mesh_device.hpp:305`) between phases is mandatory and nothing enforces
  it.

---

### DEC-028 — The one allowed raw `ttnn.all_gather` inside a module: the RMSNorm stats tensor
- **Phase / module:** P4 / `tt/rms_norm.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the plan's rule is that modules only ever call `MeshConfig` wrappers, never a
  collective op directly. `BRINGUP_RECIPE.md:921-923` grants exactly one exception — "`ttnn.all_gather`
  for the tiny RMSNorm stats tensor; if you use it, log a `DEC`" — and this package **ships that code
  path**, dormant, because it keeps the template's `is_distributed` branch.
- **Question:** does the distributed-RMSNorm branch route its stats all-gather through
  `MeshConfig.allgather`, keep the template's raw `ttnn.all_gather`, or get deleted until P8 needs it?
- **Options considered:**
  1. **Route it through `MeshConfig.allgather`.** Consistent with the rule — but the wrapper calls
     `ttnn.experimental.all_gather_async` with the ping-pong semaphores and a DRAM memory config,
     while this call needs a **specific width-sharded L1 memory config** for the
     `[1, 1, 32, 32*tp]` stats tensor that `rms_norm_post_all_gather` then consumes
     (`models/demos/gpt_oss_d_p/tt/rms_norm.py:60-65`, `:70-78`). Making the wrapper carry that would
     put a norm-specific memory config into the generic collective API.
  2. **Keep the raw `ttnn.all_gather`,** as the template does, and log it here.
  3. **Delete the branch** until scheme B is taken. Removes dead code now, and removes the seam the
     recipe asks to keep (P5.2: "keep the `is_distributed` branch ... but make it a constructor
     argument defaulting to `False` until P8").
- **Choice:** option 2. The branch stays, defaults to `is_distributed=False`, and its
  `ttnn.all_gather` is this decision's subject. It is **unreachable in this iteration**: scheme A
  (`DEC-025`) never sets the flag.
- **Why:** the exception exists in the recipe because this tensor is genuinely different — 32x256
  in L1 with a norm-specific shard spec, not a 4096-wide DRAM activation — and `ttnn.all_gather` is
  a **non-experimental** op, so it does not carry the semaphore-lifetime hazard the rule is actually
  guarding against (the ping-pong exists because `*_async` ops take explicit semaphores; this one
  does not). Option 1 would generalise the wrong thing; option 3 disobeys P5.2.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/rms_norm.py:70` (`tt_gathered_stats = ttnn.all_gather(`),
  `:74` (`cluster_axis=1` — the TP axis, consistent with §4's TP-only rule), `:60-65` (the L1
  width-sharded memory config it needs), `:67` / `:82` (the pre/post pair that brackets it);
  `BRINGUP_RECIPE.md:921-923` (the exception), `BRINGUP_RECIPE.md:1032-1033` (P5.2's instruction to
  keep the branch).
- **Confidence:** high, with the caveat that the code is dormant — a dormant branch is a claim about
  code that has never run, which is why it must not be counted as "the distributed norm works".
- **Falsifier:** P8 enabling the branch and finding the raw call races against the wrapper's
  collectives — i.e. that `ttnn.all_gather` does interact with the ping-pong state after all.
- **Revisit if:** scheme B is taken (`DEC-025`), or `G-CLEAN` decides a permanently-dormant branch is
  dead code that should go.
- **Blast radius:** `tt/rms_norm.py`; `G-RMS` (which runs only the single-pass branch), `G-TP-PARITY`,
  `G-CLEAN` items 4 and 5.

---

### DEC-029 — Drop four dead pieces of the `CCLManager` template
- **Phase / module:** P5.1 / `tt/ccl.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** porting `models/demos/gpt_oss_d_p/tt/ccl.py:17` verbatim, as `03_OUTLINE.md` §2.2
  says ("taken essentially whole"), and finding four members that no caller reads.
- **Question:** carry `_ping_pong_buffer_cache`, `_ping_pong_buffer_indices`, the
  `_worker_sub_device` local and `ccl_sub_device_id` across, or delete them?
- **Options considered:**
  1. **Carry them.** Keeps the diff against the template minimal and keeps `03_OUTLINE.md` §2.2's
     attribute list literally true, including `ccl_sub_device_id`. But agent-contract rule 5 is
     explicit: no dead code.
  2. **Delete all four.** Measured dead: `grep -rn` across `models/demos/gpt_oss_d_p/` and
     `models/demos/minimax_m3/` finds each name **only at its own definition**. The `SubDevice` is
     constructed into a local that is immediately discarded — it is never registered with the
     device — and `ttnn.SubDeviceId(0)` is then set unconditionally, so the attribute does not even
     describe the object that was built. `models/demos/deepseek_v3_d_p/tt/tt_ccl.py:67`, the
     substrate both templates say they mirror, does not create a `SubDevice` at all: it keeps a
     plain `CoreRangeSet` (`sub_device_crs`) and nothing else.
  3. **Delete the buffers, keep `ccl_sub_device_id`** in case a P8 op wants a `subdevice_id=`. The
     only in-tree call sites that take one pass `subdevice_id=None`
     (`models/demos/gpt_oss_d_p/tt/moe/tt_gpt_oss_moe.py:104`,
     `models/demos/minimax_m3/tt/moe/tt_minimax_moe.py:117`), and both are MoE, which Llama does
     not have.
- **Choice:** option 2. `self.ccl_cores` (the `CoreRangeSet` the semaphores are actually created
  on) stays; the rest go. `_init_subdevice` is renamed `_init_ccl_cores`, which is what it does.
- **Why:** rule 5, and the specific hazard that `ccl_sub_device_id` is worse than unused — it is a
  plausible-looking handle whose value is unrelated to the discarded `SubDevice`, so the first
  caller to trust it inherits a wrong id rather than an obvious `AttributeError`. Re-adding a real
  sub-device in P8, if the ring path needs one, is three lines and will be gated by the phase that
  needs it.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/ccl.py:24-25` (the two unused dicts), `:50` (the
  discarded `SubDevice` local), `:55` (`ccl_sub_device_id`);
  `models/demos/deepseek_v3_d_p/tt/tt_ccl.py:67` (the substrate's `sub_device_crs`, no `SubDevice`);
  `models/demos/gpt_oss_d_p/tt/moe/tt_gpt_oss_moe.py:104` (`subdevice_id=None`).
- **Confidence:** high on the buffers and the local; medium on `ccl_sub_device_id`, because P8's
  ring SDPA has not been written yet and could want a sub-device.
- **Falsifier:** a P8 op that requires a registered CCL sub-device, which would mean the template
  was carrying an unfinished feature rather than dead code.
- **Revisit if:** P8's `dense_sp.py` needs `subdevice_id=`; or `G-SEMAPHORE` at the target mesh
  behaves differently from the one-card run in a way that points at core-range ownership.
- **Blast radius:** `tt/ccl.py`; `03_OUTLINE.md` §2.2's attribute list (now four names shorter);
  `G-MESH`, `G-SEMAPHORE`.

---

### DEC-030 — One compute-kernel config for the package, in `tt/config.py`, `fp32_dest_acc_en=True`
- **Phase / module:** P5.1-P5.2 / `tt/config.py`, every module
- **Date (UTC):** 2026-09-04
- **Trigger:** `tt/rms_norm.py` is the first module that needs a `compute_kernel_config`, and
  recipe §2.4 requires an explicit one on **every** op that accepts one. Neither
  `BRINGUP_RECIPE.md:752-819`'s tree nor `03_OUTLINE.md` §1 gives that factory a home: the
  templates put it inside `AttentionConfig` (`models/demos/gpt_oss_d_p/tt/attention/config.py:103`),
  which only attention can reach.
- **Question:** where does the single definition live, and with which four field values?
- **Options considered:**
  1. **A new `tt/compute_config.py`.** Clear ownership; adds a file to a tree that P3 froze and
     that `G-CLEAN` audits against.
  2. **`tt/model_config.py`.** The natural home, and P6.2 owns that file — writing it in P5.1
     would step on a later phase's deliverable and force a merge.
  3. **`tt/config.py`**, beside `MeshConfig`. Already the package's config module, already imported
     by every module for the mesh, and its name does not promise "mesh only".
  4. **Per-module, inline.** What both templates do, and how
     `models/demos/gpt_oss_d_p/tt/attention/config.py:71`'s `fp32_dest_acc_en: bool = False` came to
     exist in one place and not the other.
- **Choice:** option 3 — `default_compute_kernel_config(mesh_device, *, fp32_dest_acc_en=True)`,
  with `math_fidelity=HiFi4`, `math_approx_mode=False`, `packer_l1_acc=False`. The
  `fp32_dest_acc_en` keyword exists for exactly two callers: the in-suite A/B, and P8's ring SDPA,
  where `False` is mandatory.
- **Why:** one definition is the same argument as the noise-floor helpers (`DEC-007`) — two copies
  drift, and here a drifted copy costs 96x-1168x on a matmul, not a rounding difference. The three
  non-obvious values: `HiFi4` because recipe §2.4 measures `MathFidelity` alone as a **no-op** and
  HiFi2 as marginally *worse* than no config, so the fidelity choice is free and the highest is the
  safe one; `math_approx_mode=False` matching both
  `models/demos/gpt_oss_d_p/tt/attention/config.py:70` and
  `models/common/models/llama32_1b/model.py:1026`, since an approximate SFPU is a precision
  regression this iteration has no reason to accept; `packer_l1_acc=False` because it is a
  performance knob and this iteration is functional-first (`BRINGUP_RECIPE.md:16-17`).
- **Evidence:** recipe §2.4's two A/B tables (`BRINGUP_RECIPE.md:404-410`, `:420-427`);
  `models/demos/gpt_oss_d_p/tt/attention/config.py:71` (the explicit `False` not to inherit);
  `models/demos/gpt_oss_d_p/tt/rms_norm.py:94` (the norm with no config at all);
  measured in-suite this phase, `raw/G-RMS_20260904T090144Z.log`: `fp32_dest_acc_en=True`
  0.9999957 (1.56x the floor) vs `False` 0.9999707 (10.79x) at seq 32, and 0.9999958 (1.57x) vs
  0.9999633 (13.58x) at seq 512 — a 6.90x / 8.66x error reduction on this box.
- **Confidence:** high on the flag and the home; medium on `packer_l1_acc=False`, which is
  unmeasured here.
- **Falsifier:** an op where `HiFi4` or `math_approx_mode=False` measurably costs accuracy, or
  where `packer_l1_acc=True` changes a PCC — either would mean these are not neutral defaults.
- **Revisit if:** P6.2 lands `ModelArgs` and wants to own the factory; perf work starts
  (`packer_l1_acc`); or P8 finds a second op needing `fp32_dest_acc_en=False`.
- **Blast radius:** every module's ops; `G-RMS`, `G-ROPE`, `G-MLP`, `G-ATTN`, `G-LAYER`,
  `G-MODEL`, `G-SP-RING`.

---

### DEC-031 — `rms_norm_post_all_gather` gets `stats` once: the template's dormant branch would raise
- **Phase / module:** P5.2 / `tt/rms_norm.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** copying the `is_distributed` branch from
  `models/demos/gpt_oss_d_p/tt/rms_norm.py:50-92`, which `BRINGUP_RECIPE.md:1032-1033` instructs to
  keep.
- **Question:** the template calls `ttnn.rms_norm_post_all_gather(x, tt_gathered_stats, ...,
  stats=tt_gathered_stats)` — the same tensor positionally **and** by keyword. Reproduce it, or fix
  it?
- **Options considered:**
  1. **Reproduce verbatim,** so the dormant branch stays a byte-for-byte copy of a known template.
  2. **Pass `stats` once, positionally,** and record why.
- **Choice:** option 2.
- **Why:** it is not a style question — the op's second positional parameter **is** `stats`, so the
  call cannot bind. Measured on this box:
  `ttnn.rms_norm_post_all_gather(x, s, stats=s)` raises
  `TypeError: ttnn.rms_norm_post_all_gather(): incompatible function arguments`. The branch is
  dormant in gpt-oss (`models/demos/gpt_oss_d_p/tt/rms_norm.py:33` pins `is_distributed = False`),
  so nothing has ever executed it and the bug is invisible there. Copying it would have handed P8
  a `TypeError` at the moment residual scheme B is switched on — the most expensive time to find
  it. This is the general hazard `DEC-028` names: a dormant branch is a claim about code that has
  never run.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/rms_norm.py:82` (the positional pass) and `:89`
  (`stats=tt_gathered_stats`); `models/demos/gpt_oss_d_p/tt/rms_norm.py:33` (the branch is
  dormant); the op's own signature, `Args: input_tensor, stats`, from
  `ttnn.rms_norm_post_all_gather.__doc__`; the measured `TypeError` above.
- **Confidence:** high — the failure is a hard `TypeError`, reproduced.
- **Falsifier:** none; the call either binds or it does not, and it does not.
- **Revisit if:** the op gains a distinct second positional parameter.
- **Blast radius:** `tt/rms_norm.py`'s dormant branch; P8's residual scheme B; nothing in this
  iteration's numbers. Also an upstream fix worth filing against
  `models/demos/gpt_oss_d_p/tt/rms_norm.py` (`07_RISKS.md` R-011).

---

### DEC-032 — `derive_head_dim` lives in `tt/config.py`, so `tt/rope.py` keeps the outline's signature
- **Phase / module:** P5.3 / `tt/config.py`, `tt/rope.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `tt/rope.py` needs `head_dim` and `DEC-020` says exactly one place in the package
  may derive it — but that place is `ModelArgs`, a P6.2 deliverable.
- **Question:** how does a P5.3 module get `head_dim` without either duplicating the derivation or
  writing P6.2's file early?
- **Options considered:**
  1. **Derive it inline in `tt/rope.py`.** Two derivations in the package the moment `ModelArgs`
     lands — precisely what `DEC-020` exists to prevent.
  2. **Add `head_dim` as a required keyword to every RoPE builder.** No duplication, but it changes
     three signatures away from `03_OUTLINE.md` §2.5 and pushes the derivation out to every caller,
     which is where a wrong value would then be introduced.
  3. **Put `derive_head_dim(hf)` in `tt/config.py`** and have `tt/rope.py` — and, in P6.2,
     `ModelArgs` — both call it.
- **Choice:** option 3.
- **Why:** it satisfies `DEC-020`'s actual requirement (one derivation) rather than its literal
  wording (one *file*), keeps `03_OUTLINE.md` §2.5's signatures unchanged, and puts the assertions
  that make the derivation safe — divisibility, and tile alignment of the result — in the one place
  they can be written once. P6.2's `ModelArgs.head_dim` becomes a call, not a formula.
- **Evidence:** `DEC-020`; `03_OUTLINE.md` §2.5 (the signatures kept);
  `models/demos/gpt_oss_d_p/tt/model.py:64` (`hf_config.head_dim`, the attribute Llama's config
  does not have); `00_MODEL_CARD.md` §2 (no `head_dim` key; 4096/32 = 128).
- **Confidence:** high.
- **Falsifier:** P6.2 finding it cannot call into `tt/config.py` without a circular import.
- **Revisit if:** P6.2 lands and prefers to own the derivation, in which case `tt/config.py`'s copy
  is deleted rather than kept alongside.
- **Blast radius:** `tt/config.py`, `tt/rope.py`, and P6.2's `ModelArgs`.

---

### DEC-033 — `tt/rope.py`'s public surface deviates from `03_OUTLINE.md` §2.5 in three places
- **Phase / module:** P5.3 / `tt/rope.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** writing the module against `03_OUTLINE.md` §2.5's interface block and finding it
  under-specified in two places and missing a needed knob in a third.
- **Question:** follow the outline's signatures literally, or change them and record it?
- **The three deviations, and why each:**
  1. **`build_indexed_rope` takes `chunk_size` and derives `sp` from the mesh**, where the outline
     writes `build_indexed_rope(mesh_device, hf, max_seq_len, sp, sp_axis)`. The block-cyclic
     reorder is keyed by the **per-chip chunk** (`chunk_size // sp`), so the function cannot be
     written without `chunk_size` — `models/demos/gpt_oss_d_p/tt/rope.py:115` takes it too, and
     `models/demos/deepseek_v3_d_p/tt/mla/utils.py:65` `block_cyclic_reorder` requires
     `chunk_local` as its second argument. `sp` becomes redundant once `sp_axis` is known
     (`mesh_device.shape[sp_axis]`), and taking both invites them to disagree.
     `CHUNK_SIZE` itself is still `DEC-004`'s deferral: the builder takes it as a parameter and P7
     picks the value.
  2. **`llama3_freqs(hf, seq_len, *, scaled=True)` gains a `scaled` flag.** `G-ROPE` is required to
     prove the llama3 scaling took effect (`BRINGUP_RECIPE.md:1093-1095`), which needs the unscaled
     tables for comparison. The alternative — the test re-deriving unscaled frequencies itself —
     would compare the module against a second transcription of the same formula rather than
     against the same code path with scaling off.
  3. **`rope_params(hf)` is added** as the single accessor the other four functions go through, and
     `build_prefill_rope` drops the outline's `dtype` argument.
     `models/tt_transformers/tt/common.py:534` `get_prefill_rot_mat` hard-codes `bfloat16`
     (`:542-547`) and takes no dtype, so a `dtype=` parameter that can only hold one value is a
     false promise; the assertion that would police it is worse than not offering the knob.
- **Choice:** all three, as described.
- **Why:** each is forced by a real constraint in the code being wrapped rather than by preference,
  and `03_OUTLINE.md` §2.5 was written before any of the three helpers had been called.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/rope.py:115` (`chunk_size` in the template's
  signature); `models/demos/deepseek_v3_d_p/tt/mla/utils.py:65` (`chunk_local` required);
  `models/tt_transformers/tt/common.py:534` and `:542` (bf16 hard-coded);
  `BRINGUP_RECIPE.md:1093-1095` (the scaling assertion `scaled=False` serves).
- **Confidence:** high.
- **Falsifier:** P7 finding it needs `sp` independent of the mesh shape — e.g. a chunk table built
  for a mesh other than the open one.
- **Revisit if:** P7 picks `CHUNK_SIZE` / `MAX_SEQ_LEN` (`DEC-004`), or P8's SP path needs a
  different sharding of the indexed tables.
- **Blast radius:** `tt/rope.py`; `03_OUTLINE.md` §2.5; `G-ROPE`, `G-CHUNK`, `G-CHUNK-ATTN`.

---

### DEC-034 — The `prefer-expect-error` hook matches prose, so the prose is reworded
- **Phase / module:** P5.1 / `tests/unit/test_mesh_config.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the first `pre-commit run` on the P5.1 test files failed. The offending line was not
  code: it was a **docstring sentence** explaining that the file uses `expect_error` rather than
  pytest's raises helper.
- **Question:** silence it with the documented same-line escape, or reword?
- **Options considered:**
  1. **Same-line escape.** `.pre-commit-config.yaml:55`'s pattern allows
     `# allow-pytest.raises: <why>` on the line. Inside a module docstring that renders as a stray
     comment in the prose, and it marks a line that contains no call at all — the marker would
     claim an exemption for something that never needed one.
  2. **Reword the sentence** so the literal token does not appear, and say in the file why the
     sentence is phrased the long way.
  3. **Delete the sentence.** Cheapest, and loses the one place a reader is told which fixture to
     use and why.
- **Choice:** option 2.
- **Why:** the hook is a `pygrep` over the whole file, so it cannot distinguish a call from a
  mention; that is a property of the hook, not a problem with the test. Rewording keeps the
  guidance and leaves no exemption marker that a future reader would have to evaluate.
- **Evidence:** `.pre-commit-config.yaml:51` (the hook), `:53` (`language: pygrep`), `:55` (the
  grep pattern), `:56` (the same-line override); the failing run's message named
  `models/demos/llama31_8b_d_p/tests/unit/test_mesh_config.py:20`, a docstring line.
- **Confidence:** high.
- **Falsifier:** the hook gaining comment/docstring awareness, which would make the rewording
  unnecessary.
- **Revisit if:** a later phase genuinely needs the escape for a real call the fixture cannot
  express.
- **Blast radius:** the wording of test docstrings across the package. Worth reporting upstream to
  the kit: `LANDMINES.md` describes the hook as rejecting "any `pytest.raises` in a `tests/` file"
  and does not say it also fires on the name in comments and docstrings.

---

### DEC-035 — `G-RMS` runs on **both** random and real layer-0 norm weights
- **Phase / module:** P5.2 / `tests/unit/test_rms_norm_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the recipe specifies the input distribution twice and not identically. P5.2 says
  "Drive it with **standard-normal** inputs and an **fp32** reference weight"; §1.4's worked `G-RMS`
  ledger block says "**real layer-0 norm weights**, seed 0"; and P5's per-module loop says
  "**identical random weights** driving both sides".
- **Question:** which weight source does `G-RMS` use?
- **Options considered:**
  1. **Random only.** Runs with no checkpoint, matches P5's loop, and is what makes every other P5
     gate portable. But a trained norm gain is a narrow positive distribution, nothing like
     standard normal, and §2.1's reference numbers were all measured on real weights — so a
     random-only gate cannot be compared with them.
  2. **Real only.** Comparable with §2.1, but makes the gate unrunnable on a weightless box, which
     `requires_hf_reference` exists to avoid.
  3. **Both**, as separate parametrised tests.
- **Choice:** option 3. The random-weight test is unguarded; the real-weight test carries
  `requires_hf_reference`.
- **Why:** they are materially different numeric tests, not a duplicate — measured this phase, the
  random-weight case sits at **1.54-1.57x** its floor and the real-weight case at **2.11-2.13x** a
  *different* floor (0.9999973 vs 0.9999986). Reporting one number would have hidden that the floor
  itself moves with the weight distribution. The real-weight run is also what makes the gate
  comparable to §2.1's 0.9999867/0.99995 pair, and it reproduces the recipe's expected 0.9999955 →
  measured **0.9999971** against the same 0.9999986 floor.
- **Evidence:** `BRINGUP_RECIPE.md:1037-1040` (P5.2's instruction), `BRINGUP_RECIPE.md:254-256`
  (§1.4's block naming real weights), `BRINGUP_RECIPE.md:997` (P5's "identical random weights");
  measured numbers in `raw/G-RMS_20260904T090144Z.log`.
- **Confidence:** high.
- **Falsifier:** the two weight sources landing on the same floor and the same ratio, which would
  make one of the two tests redundant.
- **Revisit if:** `G-MLP`/`G-ATTN` adopt the same two-source pattern, in which case it should be a
  package convention rather than a per-gate choice.
- **Blast radius:** `tests/unit/test_rms_norm_vs_ref.py`; `G-RMS`'s recorded numbers.

---

### DEC-036 — `G-ROPE`'s scaling control asserts the **band structure**, not beyond-window divergence
- **Phase / module:** P5.3 / `tests/unit/test_rope_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** implementing `BRINGUP_RECIPE.md:1093-1095` literally — "the scaled `inv_freq` must
  differ from the unscaled one for positions beyond `original_max_position_embeddings`" — and
  measuring that it does not discriminate.
- **Question:** keep the recipe's assertion as the control, or assert something sharper?
- **Measurement that forced the question:** `max|cos_scaled - cos_unscaled|` is **1.99933** for
  positions *inside* the original window and **1.99398** *beyond* it. Both saturate at the
  theoretical maximum of 2, because llama3 scaling divides the long-wavelength frequencies at
  **every** position and `cos` oscillates. So "the tables differ beyond 8192" is satisfied by an
  implementation that scales everything, or by one that scales nothing beyond the window — neither
  of which is llama3 scaling.
- **Options considered:**
  1. **Keep the recipe's assertion.** Literal compliance; a control that cannot fail for the reason
     it is supposed to catch, which is the exact failure mode §1.4 warns about.
  2. **Assert the piecewise band structure** directly on the frequencies `apply_scaling` consumes:
     wavelengths above `orig/low_freq_factor` divided by exactly `factor`, wavelengths below
     `orig/high_freq_factor` bit-identical (`torch.equal`), and a non-empty middle band strictly
     between the two. Measured: **29 low / 6 mid / 29 high** of 64 frequencies.
  3. Both.
- **Choice:** option 3 — the band structure is the assertion that gates, and the recipe's
  beyond-window delta is still **recorded** with both halves of the number, so a reader can see why
  it is not the discriminator.
- **Why:** a control's job is to fail when the property is broken. The band test does: disabling
  scaling makes all three bands identical to the base frequencies, and scaling everything breaks
  the `torch.equal` on the high band. A first attempt at option 1 also produced a *false failure* —
  recovering the frequency from a cos table by `arccos` gives `0/0 = nan` for the lowest frequency,
  because `cos(1 * f)` rounds to 1.0 in fp32 — which is `LANDMINES.md`'s "a failing probe is not
  evidence of a failing module until the probe's own numerics are checked", hit live. Both raw logs
  are kept.
- **Evidence:** `raw/G-ROPE_20260904T090652Z.log` (the `nan` probe failure);
  `raw/G-ROPE_20260904T091040Z.log` (the band counts and both deltas);
  `models/tt_transformers/tt/common.py:405` `compute_llama3_parameters` (the three branches being
  asserted), `:407-408` (the factors).
- **Confidence:** high.
- **Falsifier:** a checkpoint whose `low_freq_factor`/`high_freq_factor` leave one band empty,
  which `assert_llama3_factors` would catch first (`07_RISKS.md` R-010).
- **Revisit if:** long-context `G-MODEL` shows the "good at short seq, bad past ~8192" symptom
  anyway, which would mean the band test passes while something downstream of the tables is wrong.
- **Blast radius:** `tests/unit/test_rope_vs_ref.py`; `G-ROPE`'s negative-control field. The recipe
  sentence itself is reported as a defect rather than silently followed.

---

### DEC-037 — Re-include `bringup_log/raw/*.log` in git, which the root `.gitignore` excludes
- **Phase / module:** P5.1 / `bringup_log/raw/.gitignore`
- **Date (UTC):** 2026-09-04
- **Trigger:** `git status` after writing the first device gate's raw log showed **nothing** under
  `bringup_log/raw/`. `git check-ignore -v` names `.gitignore:7`, the repo-root `*.log` rule. So
  every raw log this bring-up has produced since P0 — `G-CARD`, `G-REF`, `G-SURVEY`, `G-OUTLINE`,
  `G-CCL-PLAN`, and now the P5 gates — exists **on this disk only** and has never been tracked
  (`git ls-files bringup_log/raw/` is empty, `git log -- bringup_log/raw/` is empty, across three
  prior committed phases).
- **Question:** leave the raw logs untracked, rename them out of the ignore pattern, or re-include
  them?
- **Options considered:**
  1. **Leave them untracked.** Contradicts the recipe's own central rule — "A gate with no raw log
     did not happen" (`BRINGUP_RECIPE.md:199`) — and Appendix C item 2, which requires every gate
     "recorded in `bringup_log/06_GATES.md` **with raw logs**". A fresh clone of this branch would
     contain a ledger citing 15 files that do not exist, i.e. the evidence base advertised by the
     deliverable would be absent.
  2. **Rename them** to `.txt` or extensionless. Escapes the pattern, and invalidates every
     `raw/<GATE>_<ts>.log` citation already written into `06_GATES.md` by four completed phases —
     the rename hazard §0.2 spends five paragraphs warning about, for no gain.
  3. **A nested `.gitignore` in `bringup_log/raw/` containing `!*.log`.** A deeper pattern wins
     over a shallower one, so the logs become trackable and nothing outside the package changes.
     Verified: `git status -uall bringup_log/raw/` went from one ignored directory to 14 log files
     plus the new `.gitignore`.
- **Choice:** option 3. Two lines of pattern plus a comment explaining why, inside the package.
- **Why:** it is the only option that satisfies the recipe without touching anything outside
  `models/demos/llama31_8b_d_p/` and without rewriting citations. It also fixes the defect
  *retroactively* for P0-P4's logs, which are still on disk. The alternative reading — that raw
  logs are deliberately local scratch — is contradicted by `LANDMINES.md`'s own advice to **gzip**
  an oversized raw log so it passes the `check-large-files` **commit** hook: that advice only makes
  sense if the logs are meant to be committed, which under the root ignore they cannot be.
- **Evidence:** `.gitignore:7` (`*.log`); `BRINGUP_RECIPE.md:199` (the rule);
  `BRINGUP_RECIPE.md:1840-1842` (Appendix C item 2); `LANDMINES.md`'s `check-large-files` row
  (gzip rather than trim, "so the evidence stays byte-exact"); `git check-ignore -v` and
  `git ls-files` output above.
- **Confidence:** high on the diagnosis; medium on the remedy being the one a repo maintainer would
  choose — an alternative is a root-level exception for `bringup_log/`, which this session may not
  write.
- **Falsifier:** a maintainer stating that bring-up raw logs are intentionally local and the ledger
  should cite them as reproduction commands rather than as artefacts — in which case the recipe's
  "a gate with no raw log did not happen" needs rewording, not this file.
- **Revisit if:** the kit adds guidance on this, or a root `.gitignore` exception lands.
- **Blast radius:** the size of the package's commits (14 logs, ~180 KB today, all well under the
  500 KB hook limit); `G-CLEAN`'s file inventory, which gains one `.gitignore`; nothing numeric.

---

### DEC-038 — `MLP(scatter_output=True)` refuses rather than running scheme B's tail
- **Phase / module:** P5.4 / `tt/mlp.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:992-994` requires the `scatter_output` seam wired "from day one",
  and in the same sentence requires any module that "cannot honour it" to **refuse** loudly. For the
  MLP specifically both halves are technically available — `MeshConfig.reduce_scatter` exists
  (`tt/config.py`) and the template implements the branch
  (`models/demos/minimax_m3/tt/dense_mlp.py:100-109`) — so the question is not capability.
- **Question:** wire the reduce-scatter branch (it is six lines and it would run), or accept the
  parameter and raise on `True`?
- **Options considered:**
  1. **Implement the branch.** `MLP` would return `[1,1,S_loc,512]` on `scatter_output=True`. Cheap,
     matches the template exactly, and it would go **untested**: `G-MLP` runs at `(1,1)` where
     `tp == 1` and the whole tail is skipped, so the branch would ship as code that has never
     executed on this box — the same "dormant branch" state that produced the
     `rms_norm_post_all_gather` double-`stats` bug (`DEC-031`).
  2. **Accept and refuse.** The parameter exists, the derived value is named
     (`_SCHEME_A_SCATTER_OUTPUT`), and `True` raises `NotImplementedError` naming `DEC-025` and P8.
  3. **Do not accept the parameter at all.** Cleanest today, and it makes the scheme-B switch a
     signature change in every module rather than a flag — exactly what the recipe says to avoid.
- **Choice:** option 2.
- **Why:** a `True` that reduce-scatters *only in the MLP* is not scheme B, it is a mixed residual —
  attention still all-reduces to full emb (`bringup_log/04_CCL_PLAN.md` §5 rows 1 and 3), the norms
  still expect full emb, so the layer's second residual add would put a 512-wide tensor against a
  4096-wide stream. That is the "half-wired scheme" the recipe's refusal clause is about. The
  refusal is *testable* today (`test_mlp_refuses_scatter_output`), whereas the branch is not, and a
  refusal that fires is better evidence than a branch that has never run.
- **Evidence:** `BRINGUP_RECIPE.md:992-994` (wire the seam, refuse what you cannot honour);
  `DEC-025` (scheme A, and `scatter_output` is its seam); `bringup_log/04_CCL_PLAN.md` §5 row 4
  ("scheme B seam; **refuses** until P8"); `models/demos/minimax_m3/tt/dense_mlp.py:100-109` (the
  branch not taken); `DEC-031` (what a dormant copied branch cost the last time).
- **Confidence:** high.
- **Falsifier:** P8 enabling scheme B and finding that the MLP was the only module that needed
  changing — i.e. that the mixed-residual objection was wrong. The other modules' tails
  (`bringup_log/04_CCL_PLAN.md` §5 rows 1, 3, 6) are what make it right.
- **Revisit if:** P8 wires scheme B, or long-context DRAM pressure forces it earlier
  (`DEC-025`'s own falsifier).
- **Blast radius:** `tt/mlp.py`, `tests/unit/test_mlp_vs_ref.py`; the same decision is owed for
  `attention/operations.apply_reduce_scatter` in P5.5 (`DEC-041`).

---

### DEC-039 — SwiGLU spelling: the fused `input_tensor_a_activations` unary, measured bit-for-bit against `ttnn.silu`
- **Phase / module:** P5.4 / `tt/mlp.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:1142-1143` offers two spellings for the activation — "`ttnn.silu(gate) * up`,
  or `ttnn.mul(..., input_tensor_a_activations=[ttnn.UnaryOpType.SILU])` **if available — check, and
  log which**".
- **Question:** is the fused unary available on this build, and if so does it cost accuracy?
- **Options considered:**
  1. **`ttnn.silu(gate)` then `ttnn.mul`.** Two ops, one extra full-size `[1,1,S,1792]` intermediate.
  2. **Fused:** `ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])`. One op.
- **Checked, because "if available" needed an answer rather than an assumption:** the keyword is
  **bound and working** — `ttnn.mul(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])`
  returns a tensor on this box — but it is **not in `ttnn.mul.__doc__`**. `multiply` is registered
  through `bind_binary_operation_with_fast_approx`
  (`ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp:2082`), whose argument list does
  include `input_tensor_a_activations` (`:1469`) while its generated docstring lists only
  `fast_and_approximate_mode`, `memory_config` and `output_tensor`. So `hasattr`/docstring probing
  says "unavailable" and the call says "available"; only the call is right.
  `ttnn.UnaryOpType.SILU` exists. The alias `activations=` also accepts it but applies the unary to
  the **output**, which is a different function — do not reach for it.
- **Choice:** the **fused** form, default `fused_silu=True`, with the separate spelling kept as an
  argument so `G-MLP` measures both.
- **Why:** measured on this box at seq 512, the two spellings are **numerically identical** —
  bf8_b `0.9999144` vs `0.9999144`, bf16 `0.9999852` vs `0.9999852`, both at 1.10x / 2.10x of the
  same floor — so the choice is free on accuracy and the fused form wins on op count and on one
  fewer live `[1,1,S,1792]` intermediate at 4096 tokens. Recording it as a `DEC` is the point: the
  recipe asked which, and "they are the same number" is the answer.
- **Evidence:** `raw/G-MLP_20260904T093653Z.log` (`[G-MLP] SiLU spelling` lines, both dtypes);
  `ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp:1469` (the binding), `:2082`
  (`multiply` registered through that binder); `BRINGUP_RECIPE.md:1142-1143`.
- **Confidence:** high — this is a measurement, not a judgement.
- **Falsifier:** a shape or dtype where the two spellings diverge, or a `ttnn` release that drops
  the undocumented keyword (which is the real risk of depending on it — hence the retained
  `fused_silu=False` path rather than deleting it).
- **Revisit if:** the keyword is removed or documented differently, or a perf pass measures the op
  count difference and finds it does not matter.
- **Blast radius:** `tt/mlp.py`'s `__call__` only. The negative control
  (`test_mlp_silu_on_wrong_branch_negative_control`) is what keeps either spelling honest about
  *which* argument the unary lands on.

---

### DEC-040 — `ProgramConfig` delegates the compute-kernel config instead of holding its own four fields
- **Phase / module:** P5.5 / `tt/attention/config.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `bringup_log/03_OUTLINE.md` §2.7 pins `ProgramConfig` with four compute fields
  (`math_fidelity: str = "HiFi4"`, `math_approx_mode`, `fp32_dest_acc_en`, `packer_l1_acc`) and a
  no-argument `get_compute_kernel_config(self)`, copying
  `models/demos/gpt_oss_d_p/tt/attention/config.py:69-72` and `:102-108`. That is a **second**
  definition of the thing `DEC-030` made single.
- **Question:** keep the outline's four local fields, or call
  `tt/config.py::default_compute_kernel_config`?
- **Options considered:**
  1. **The outline's signature verbatim.** Matches the template and needs no device handle. It also
     recreates precisely the condition `BRINGUP_RECIPE.md:1032-1035` blames: "Give it one, reachable
     home here rather than burying it inside an attention config — the only in-repo precedent
     buries it, which is plausibly how one package ended up with `fp32_dest_acc_en=False` in one
     place and the correct value in another."
  2. **Delegate entirely**, no local fields. Then `G-ATTN` cannot A/B §2.4's block-level 38.7x /
     107.6x claim, which the recipe asks for ("A/B it in-suite so a regression shows up as a number
     rather than a mystery").
  3. **Delegate, keeping `fp32_dest_acc_en` as the one local field**, and take `mesh_device` as an
     argument because the factory needs the arch.
- **Choice:** option 3. `get_compute_kernel_config(mesh_device)` returns
  `default_compute_kernel_config(mesh_device, fp32_dest_acc_en=self.fp32_dest_acc_en)`.
- **Why:** three of the four fields had exactly one correct value and no gate needed them variable,
  so holding them locally could only ever create a second place for them to be wrong. The fourth is
  the one recipe §2.4 says to measure, so it stays — as a measurement knob with a docstring saying
  so, the same shape `tt/rms_norm.py` and `tt/mlp.py` already use. The signature change
  (`self` -> `self, mesh_device`) is the cost, and it is paid once at each call site inside
  `attention/prefill.py`.
- **Evidence:** `BRINGUP_RECIPE.md:1032-1035` (one reachable home, and the diagnosis of the
  precedent); `models/demos/gpt_oss_d_p/tt/attention/config.py:102-108` (the buried copy);
  `DEC-030`; `bringup_log/03_OUTLINE.md` §2.7 (the signature deviated from); the `G-MLP` A/B
  measuring 96.13x / 1167.80x for `False` on this box, which is why the knob is worth keeping.
- **Confidence:** high.
- **Falsifier:** a per-op need for a different `math_fidelity` inside attention — e.g. a
  measurement showing LoFi is enough for the SDPA QK matmul — which would want a second factory
  argument rather than four resurrected fields.
- **Revisit if:** P6.2's `ModelArgs` takes over the compute-config home (`DEC-030`'s own open
  question), or a perf pass wants per-op fidelity.
- **Blast radius:** `tt/attention/config.py`, `tt/attention/prefill.py`,
  `tests/unit/test_attention_vs_ref.py`; `bringup_log/03_OUTLINE.md` §2.7's stated signature is now
  out of date and this entry is the record of why.

---

### DEC-041 — attention's `apply_reduce_scatter` refuses, for the same reason the MLP's does
- **Phase / module:** P5.5 / `tt/attention/operations.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `bringup_log/04_CCL_PLAN.md` §5 row 3 lists this call site as "scheme B seam;
  **refuses** until P8", and `bringup_log/03_OUTLINE.md` §2.7 lists the function in
  `operations.py`'s interface.
- **Question:** same question as `DEC-038`, for the other module that closes with a TP collective.
- **Choice:** the function exists, is named in `operations.py`, and raises `NotImplementedError`
  naming `DEC-025`, P8 and the CCL-plan row.
- **Why:** identical to `DEC-038` — a reduce-scatter in attention while the norms, the residual add
  and the MLP all still work in full emb is a mixed residual, not scheme B. Recording it as its own
  entry rather than folding it into `DEC-038` because the blast radii differ: the attention seam is
  also what P8's scheme-B work has to touch alongside the ring path, and `04_CCL_PLAN.md` numbers
  the two call sites separately (rows 3 and 4).
- **Evidence:** `DEC-038`; `DEC-025`; `bringup_log/04_CCL_PLAN.md` §5 rows 3-4;
  `BRINGUP_RECIPE.md:992-994`.
- **Confidence:** high.
- **Falsifier:** as `DEC-038` — P8 finding that a per-module switch is coherent after all.
- **Revisit if:** P8 wires scheme B.
- **Blast radius:** `tt/attention/operations.py`. Note that unlike the MLP's, this refusal is
  **not** covered by a test: nothing calls it, so a test would be asserting that a function this
  package never invokes raises. Recorded here rather than papered over with a test that proves
  nothing about the model.

---

### DEC-042 — `G-ATTN`'s 8x block budget: hold it at bf8_b, gate bf16 on the SDPA-attributed residual
- **Phase / module:** P5.5 / `tests/unit/test_attention_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:1793` sets `G-ATTN` at "PCC >= 0.999; own stages <= 3x floor,
  block <= 8x". Measured on this box, the block clears 0.999 at both dtypes and clears 8x at
  **bf8_b** but **not** at bf16:

  | dtype | S | block PCC | block floor | raw ratio | 8x? |
  |---|---|---|---|---|---|
  | bf8_b | 128 / 512 / 2048 | 0.9997364 / 0.9997080 / 0.9996723 | 0.9998805 / 0.9998657 / 0.9998521 | **2.21x / 2.17x / 2.22x** | yes |
  | bf16 | 128 / 512 / 2048 | 0.9998463 / 0.9998275 / 0.9998029 | 0.9999875 / 0.9999854 / 0.9999838 | **12.32x / 11.82x / 12.17x** | **no** |

  Note the direction: bf16 has the **higher** absolute PCC and the **worse** ratio, because a
  smaller floor error makes the same fixed slack a larger multiple.
- **Question:** is the module wrong, is the threshold wrong, or is the *metric* being applied to
  something the recipe says it does not describe?
- **The attribution, measured rather than argued.** Recipe §2.3: "the floor model ... does not
  describe a **fused** kernel's interior ... So do not read a large block-level gap as 'our code is
  wrong' before isolating the fused kernel." Isolated in the same test, on the same tensors:
  - every stage this package implements sits at **1.00x-2.50x** of its own **locally computed**
    floor, at both dtypes (recipe's own run: 1.00-1.47x);
  - `ttnn.transformer.scaled_dot_product_attention` alone sits at **26.7x-28.6x** of its modelled
    floor on the block's own post-RoPE tensors, and at **52.8x-55.0x** on iid standard-normal Q/K/V
    (recipe's own run: 71x);
  - `1 - PCC` is variance-like, so independent error sources add to first order. Subtracting the
    fused kernel's own excess from the block error predicts the block PCC to 5-6 decimal places —
    bf8_b S=512: predicted **0.9997079**, measured **0.9997080**; bf16 S=512: predicted
    **0.9998277**, measured **0.9998275** — and leaves an **SDPA-attributed residual ratio of
    0.70x-1.10x** across all six cases. That residual is this package's code, and it is *at* the
    floor.

  So the excess is **entirely** the fused kernel, quantitatively, not by assertion. And the 8x block
  budget is arithmetically unreachable at bf16 given that kernel: 8x would need the kernel's excess
  under `7 x 1.46e-5 = 1.02e-4`, i.e. the kernel at <= ~17x its own floor. It measures 26.7x here
  and the recipe measured 71x. **A block budget of 8x at bf16 is inconsistent with the recipe's own
  §2.3 measurement of the same kernel** — the two numbers cannot both be met by any correct
  implementation.
- **Options considered:**
  1. **Assert 8x at both dtypes.** The gate FAILs, which under §0 rule 1 stops the whole bring-up,
     on a module whose every hand-written stage is at its floor. Wrong answer to a metric problem.
  2. **Raise the budget to 13x.** Fitting a threshold to a measurement already seen — the recipe's
     own named error, "the same error with a friendlier face" (`BRINGUP_RECIPE.md:1838-1840`).
  3. **Drop bf16 from the gate.** Loses the measurement the recipe asks for and hides the finding.
  4. **Keep 0.999 and the 3x stage budgets at both dtypes; assert the raw 8x where it holds
     (bf8_b, the package's weight dtype); and at both dtypes assert the SDPA-attributed
     residual <= 8x**, logging the raw ratio, the kernel's excess and the additive prediction every
     run.
- **Choice:** option 4.
- **Why:** it is the recipe's own sanctioned handling in §2.3 — "separate error budgets per stage,
  measured rather than assumed, plus a permanent standalone probe of the fused kernel so its slack
  is named and tracked" — expressed as an assertion instead of a paragraph. Nothing is loosened: the
  residual assertion is **tighter** than the raw 8x (measured 0.70x-1.10x against a budget of 8x),
  so a real regression in any stage this package wrote still trips it at bf16, which is exactly what
  §2.3 warns a lumped budget would absorb. The raw ratio is recorded at both dtypes so the finding
  is visible rather than defined away.
- **Verdict recorded:** `PASS-WITH-DEVIATION`, because the literal Appendix A wording is not met at
  bf16 and this entry is the deviation.
- **Evidence:** `raw/G-ATTN_20260904T095359Z.log` (all six block cases with their attribution
  lines, the eight stage lines per dtype, and the three standalone-probe lines);
  `BRINGUP_RECIPE.md:1793` (the threshold), `:396-412` (§2.3, the fused-kernel caveat and the
  sanctioned handling), `:1799-1802` (do not refit a threshold after seeing the number).
- **Confidence:** high on the attribution (it is a measurement that predicts to 5 decimals); medium
  on the remedy being what the recipe's author would choose — the alternative reading is that the
  8x budget was only ever meant for bf8_b and the bf16 row simply is not gated on a ratio, which is
  the same thing this entry does with the residual assertion added.
- **Falsifier:** the SDPA kernel improving (or being replaced) such that the raw bf16 ratio drops
  under 8x — at which point `RAW_BLOCK_BUDGET_APPLIES[bfloat16]` should flip to `True` and this
  entry retires. Equally falsifiable the other way: if the residual ratio ever exceeds ~2x while the
  kernel's excess is unchanged, the additive attribution is wrong and the block gap is not the
  kernel's.
- **Revisit if:** the SDPA kernel changes; `G-LAYER` / `G-MODEL` (whose budgets are 8x and 4x) hit
  the same wall, in which case this is a recipe-wide issue rather than a `G-ATTN` one.
- **Blast radius:** `tests/unit/test_attention_vs_ref.py`'s block assertions;
  `bringup_log/06_GATES.md`'s `G-ATTN` verdict; the same arithmetic will apply to `G-LAYER` and
  `G-MODEL` in P6, since both contain this kernel.

---

### DEC-043 — Correction to `DEC-019`'s evidence: the head-split op's keyword is `num_heads`, not `num_q_heads`
- **Phase / module:** P5.5 / `tt/attention/operations.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** implementing `split_qkv_heads_prefill` from `DEC-019` /
  `bringup_log/03_OUTLINE.md` §2.7, both of which spell the call as
  `ttnn.experimental.nlp_create_qkv_heads(q, ttnn.concat([k, v], dim=3), num_q_heads=4, num_kv_heads=1, transpose_k_heads=False)`.
- **What is wrong:** there is no `num_q_heads` keyword. The binding is
  `nb::arg("input")`, `nb::arg("input_kv")`, `nb::arg("num_heads")`, `nb::arg("num_kv_heads")`,
  `nb::arg("transpose_k_heads")`, `nb::arg("kv_tied")`, `nb::arg("memory_config")`,
  `nb::arg("output_tensors")` —
  `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_nanobind.cpp:27-35`.
  The C++ *function* parameter is named `num_q_heads`
  (`.../nlp_create_qkv_heads.cpp:12`), which is where the outline's spelling came from; the Python
  keyword is not. Written as documented it is a `TypeError` on the first call — loud, so it cost
  minutes rather than a session, but it means `DEC-019`'s "evidence" was never executed.
- **Choice:** call it `num_heads=`, keep everything else `DEC-019` decided (the two-tensor
  `(q, cat(k, v))` form, `transpose_k_heads=False`), and record the correction rather than editing
  `DEC-019` — the log is append-only.
- **Why this is worth an entry at all:** `DEC-019` presents the call as verified against a
  `path:line`, and the citation is to the C++ overload rather than to the binding. A citation that
  resolves to a real line and still does not support the claim is the failure mode §1.6 exists to
  catch, and `verify_citations.py` cannot catch this one — the line exists and contains the string.
- **Evidence:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_nanobind.cpp:30`
  (`nb::arg("num_heads")`); `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp:12`
  (the C++ parameter that is named `num_q_heads`); `raw/G-ATTN_20260904T095359Z.log` (the working
  call, in the head-split stage lines).
- **Confidence:** high — the call runs.
- **Falsifier:** none; it is an API fact.
- **Revisit if:** the op's binding is renamed.
- **Blast radius:** `tt/attention/operations.py`; `bringup_log/03_OUTLINE.md` §2.7's stated call;
  `DEC-019`'s evidence line.

---

### DEC-044 — The probe's exact-integer ceiling is **128** at `bfloat8_b`, not §2.5's 256
- **Phase / module:** P5.6 / `tests/unit/test_kv_cache_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `G-KV`'s bit-exact positional read-back, built to `BRINGUP_RECIPE.md:1260-1262`'s
  stated rule — "encode positions as values **<= 256** or bf16 rounds 257 to 256 and the probe fails
  on a correct cache (§2.5)" — with 4 chunks of 64 covering positions 0..255. It **failed** at
  `bfloat8_b`: `chunk 2 is not bit-identical ... (max|delta| = 1.0); rows [1, 3, 5, 7, 9, 11, 13, 15]`
  — the **odd** positions of the 128..191 range.
- **Question:** is the cache wrong, or is the probe?
- **The probe. Measured, not reasoned:** quantising the integers 0..511 through
  `quantize_like_device` with each 16-lane block held **constant** (so the shared exponent is
  already ideal) gives a first inexact integer of **129** at `bfloat8_b` and **257** at
  `bfloat16`. §2.5's 256 is the **bf16** ceiling; `bfloat8_b`'s is **128**. Above it, bf8_b's
  7-bit-per-block magnitude resolution is 2, so odd integers round to their even neighbour —
  exactly the observed `max|delta| = 1.0` on odd rows.
- **Why this is not a footnote:** §2.5's rule is stated in the same recipe that mandates
  `bfloat8_b` as the KV cache dtype (P5.6, "every threshold in Appendix A assumes it"), so a probe
  written to the stated ceiling is guaranteed to fail on the very dtype the gate is supposed to
  measure. And it fails in the shape §2.5 itself warns about — "A failing probe is **not** evidence
  of a failing module until the probe's own numerics are checked". The trap caught its own author's
  fix.
- **Options considered:**
  1. **Run the bit-exact probe only at bf16.** Leaves the deployment dtype's addressing ungated.
  2. **Split the position id across lanes** (§2.5's other suggested fix): low 7 bits in one lane
     block, high bits in another. Works, and doubles the probe's own arithmetic — more code that can
     be wrong in a test whose whole job is to be obviously right.
  3. **Halve the chunk: 4 chunks of 32 = 128 positions, ids 0..127.** Keeps 4 distinct `kv_actual`
     offsets {0, 32, 64, 96} (all tile-aligned), keeps the whole probe inside **both** dtypes'
     exact range, and needs no encoding.
- **Choice:** option 3, with the per-dtype ceiling recorded in the test as a named, measured
  constant (`EXACT_INTEGER_CEILING = {bfloat8_b: 128, bfloat16: 256}`) and asserted against the
  payload, so the next person to widen the probe is stopped by an assertion rather than by a
  mysterious `max|delta| = 1.0`.
- **What it costs:** the probe covers 128 global positions instead of 256. The pad tail grows from
  128 to 256 untouched positions, so the "no collateral writes" half gets *stronger*, and the
  `kv_actual` coverage (4 offsets) is unchanged — which is the constraint §2.5 actually cared about.
- **Evidence:** `raw/G-KV_20260904T100312Z.log` (both dtypes bit-identical, ceiling logged per
  dtype); the measurement above, run on this box; `BRINGUP_RECIPE.md:1260-1262` (the rule as
  stated), `:477-487` (§2.5).
- **Confidence:** high — it is a measured property of the dtype.
- **Falsifier:** a `bfloat8_b` implementation with 8-bit block mantissas, which would move the
  ceiling to 256 and make §2.5's rule correct for both dtypes.
- **Revisit if:** the probe needs to cover more than 128 positions (then option 2, split lanes), or
  `G-KV-TP8` needs global positions past 128 — which it will, at `(1,8)` with a real chunk size, and
  it must use the split-lane encoding rather than raising the ceiling.
- **Blast radius:** `tests/unit/test_kv_cache_vs_ref.py`; `G-KV-TP8`'s probe design (P8);
  `G-KV-TABLE`'s (P10) — every bit-exact probe over a bf8_b tensor in this package.

---

### DEC-045 — `expect_error`'s `message` is a regex, so refusal messages are matched on a metachar-free substring
- **Phase / module:** P5.6 / `tests/unit/test_kv_cache_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `test_kv_cache_refuses_unaligned_capacity` failed with
  `AssertionError: Regex pattern did not match` while the assertion it was checking had fired
  correctly with the expected text. The message asked for was
  `"must be a multiple of TILE_SIZE*sp"`, which as a **regex** means `TILE_SIZ` followed by zero or
  more `E` followed by `sp` — and therefore does not match the literal string it was copied from.
- **Question:** change the module's message, or the test's matcher?
- **What is actually going on:** the repo-root `expect_error` fixture (`conftest.py:948`) documents
  its argument as prose — "`message` must appear in the real device error text (the TT_FATAL line),
  since that's what the triager matches" — which reads as a substring check. It is implemented as a
  regex match, so any assertion message containing `*`, `(`, `)`, `[`, `.` or `+` is a live trap.
  Assertion and `TT_FATAL` text is full of parenthesised values (`kv_actual (16) must be...`), so
  this will recur for every refusal test in this package.
- **Choice:** match on a **metachar-free substring** of the real message
  (`"seq_local must be tile-aligned"`), and note the reason at the call site. The module's message
  is not changed — it is the more useful text for a human reader, and rewording device-facing errors
  to suit a test matcher is the wrong direction.
- **Why not escape it instead:** `re.escape` at the call site would work, and would look like
  ordinary noise to the next reader; a one-line comment naming the cause is what stops the same
  hour being spent twice. The convention for this package is therefore: **pick a substring with no
  regex metacharacters**, which every message in `tt/` has.
- **Evidence:** `conftest.py:948` (the fixture and its substring-implying docstring), `:962` (the
  match that raised `Regex pattern did not match`); the failing run's own output;
  `LANDMINES.md`'s `prefer-expect-error` row, which covers the hook but not this.
- **Confidence:** high.
- **Falsifier:** the fixture being changed to a literal substring check upstream, which would make
  the whole entry moot (and would be the better fix).
- **Revisit if:** a refusal's only distinctive text contains a metacharacter — then escape, and say
  so.
- **Blast radius:** every `expect_error` call in this package —
  `tests/unit/test_{mesh_config,mlp_vs_ref,attention_vs_ref,kv_cache_vs_ref}.py` today, plus
  `G-RUNTIME`'s nine refusals and `G-SP-RING`'s `TT_FATAL` in later phases. Worth a kit note: this
  is a repo-wide trap, not a Llama one.

### DEC-046 — `ModelArgs` does **not** apply `map_hf_to_meta_keys`; the package's key convention is HF names
- **Phase / module:** P6.2 / `tt/model_config.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** recipe P6.2 specifies the loader as "state-dict loading (`load_state_dict` via
  safetensors, then `map_hf_to_meta_keys` / `convert_hf_qkv_to_meta_format` from
  `models/tt_transformers/tt/load_checkpoints.py`)" (`BRINGUP_RECIPE.md:1397`). Applying that map to
  this package's state dict makes **every** module weight go missing.
- **Question:** does `ModelArgs.load_state_dict` rename the checkpoint's keys into Meta form, as the
  recipe says, or leave them in HF form, as every module written in P5 expects?
- **The conflict, concretely.** `map_hf_to_meta_keys`
  (`models/tt_transformers/tt/load_checkpoints.py:800`) rewrites `model.` -> ``,
  `self_attn` -> `attention`, `q_proj` -> `wq`, `mlp` -> `feed_forward`, `gate_proj` -> `w1`,
  `input_layernorm` -> `attention_norm`, `lm_head` -> `output`, `embed_tokens` -> `tok_embeddings`.
  The P5 modules split on the **HF** names: `tt/attention/weights.py` reads
  `substate(state_dict, "q_proj")["weight"]`, `tt/mlp.py` reads `"gate_proj"` / `"up_proj"` /
  `"down_proj"`, and `tt/layer.py` splits `"self_attn"` / `"mlp"` / `"input_layernorm"` /
  `"post_attention_layernorm"`. Measured: after the map, **0 of 291** checkpoint keys are names this
  package consumes.
- **Options considered:**
  1. **Apply the map and rewrite the P5 modules to Meta names.** Follows the recipe literally. Costs
     a rename across five gated modules and seven gated test files, re-running six P5 gates, for no
     numerical change — and it would make the package's key names differ from the HF anchor each
     module's docstring names, which is the thing the docstring-anchor convention exists to prevent.
  2. **Apply the map and translate back inside each module.** Two naming systems, one of them
     invisible. This is the "silent mix" recipe P1 trap 2 exists to forbid, one level up.
  3. **Do not apply it.** The checkpoint's keys are already `model.layers.N.self_attn.q_proj.weight`
     — exactly what the modules split — so the loader's key transform is the identity, and
     `expected_state_dict_keys()` states the contract explicitly instead of leaving it implicit.
- **Choice:** option 3, and the recipe's negative control is **inverted rather than dropped**.
- **Why:** `map_hf_to_meta_keys` exists for `models/tt_transformers`, which loads Meta-format
  checkpoints and whose modules are named `wq`/`w1`/`attention_norm`. This package loads an **HF**
  checkpoint into modules whose docstrings anchor to `transformers.models.llama.modeling_llama`, so
  a Meta rename would be a translation into a convention nothing here uses. The recipe's control —
  "bypass `map_hf_to_meta_keys` and every key must go missing" (`BRINGUP_RECIPE.md:1409`) — is
  sound in intent (prove the loader is name-sensitive) and inapplicable as written (there is nothing
  to bypass). `G-WEIGHTS` therefore **applies** the map and requires every expected key to go
  missing, which discriminates identically.
- **Evidence:** `map_hf_to_meta_keys`' replacement table
  (`models/tt_transformers/tt/load_checkpoints.py:800`); measured at `G-WEIGHTS`
  (`test_meta_key_mapping_negative_control`): 291 keys in, **0** still consumable, first three
  `layers.0.attention.wk.weight` / `wo` / `wq`; and `test_model_refuses_meta_renamed_state_dict`,
  where a Meta-renamed dict makes `Model` raise rather than build on `None`s.
- **Confidence:** high.
- **Falsifier:** a module in this package that reads a Meta-form key. `G-CLEAN` can grep for
  `wq`/`w1`/`attention_norm` across `tt/` to keep it that way.
- **Revisit if:** the package is ever pointed at a Meta-format checkpoint (then the map belongs in
  `load_state_dict`, behind an explicit format argument), or P10's adapter is handed a state dict by
  the engine in Meta form.
- **Blast radius:** `tt/model_config.py`, every module's `state_dict` contract, `G-WEIGHTS`'s
  negative control.

---

### DEC-047 — `load_state_dict(convert_to_meta_format=True)` refuses: the Q/K swizzle has exactly one home
- **Phase / module:** P6.2 / `tt/model_config.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `bringup_log/03_OUTLINE.md` §2.3 pins the signature as
  `load_state_dict(weights_path, dummy_weights=False, convert_to_meta_format=True)`, copied from
  `models/demos/gpt_oss_d_p/tt/model_config.py:106`, which **does** call
  `convert_hf_qkv_to_meta_format` there. But `tt/attention/weights.py` already calls it at load
  (`DEC-011`, P5.5), for the stated reason that a weight must not be able to reach the device
  un-swizzled via a path that forgot.
- **Question:** where does the HF->Meta Q/K `reverse_permute` happen — in the checkpoint loader, in
  the attention weight loader, or (the failure) both?
- **Options considered:**
  1. **Loader only** (the gpt-oss arrangement). Then `tt/attention/weights.py` must stop swizzling,
     and any caller that builds `Attention` from a raw checkpoint slice — every P5 unit test does —
     silently gets an HF-layout weight and a RoPE that is wrong at every position.
  2. **Attention only** (the status quo since P5.5). One home, and it is the one on the path every
     weight takes.
  3. **Both**, i.e. keep the outline's default. `reverse_permute` applied twice is **not** the
     identity: it is a different permutation of the head dim, so Q/K reach the device in a third
     layout. The device runs, the PCC is plausible, and nothing raises.
- **Choice:** option 2, with the parameter **kept in the signature and refused when `True`**.
- **Why:** keeping the parameter preserves the outline's contract for a P7/P10 caller that copied it
  from the template, and refusing turns option 3 from a silent wrongness into a `NotImplementedError`
  naming the reason. Deleting the parameter would make the same mistake a `TypeError` at a random
  call site instead of an explained refusal at the one place the decision lives.
- **Evidence:** `models/tt_transformers/tt/load_checkpoints.py:451` matches on the key substrings
  `"q_proj.weight"` / `"k_proj.weight"`, so a second application hits the same two tensors;
  `reverse_permute` (`:891`) and `permute` (`:895`) are inverses, and applying `reverse_permute`
  twice equals neither. Measured at `G-WEIGHTS` (`test_double_meta_swizzle_is_caught`): with Q/K
  pre-swizzled, `q_proj` and `k_proj` stop being bit-equal to the clean load while `v_proj` and
  `o_proj` are untouched — i.e. the bit-exact check sees it, which is why part (c) of that gate is
  bit-equality and not PCC.
- **Confidence:** high.
- **Falsifier:** a `G-ATTN` or `G-MODEL` PCC that improves when the loader also swizzles.
- **Revisit if:** the package moves to `ttnn.experimental.rotary_embedding_hf`, which removes the
  permute entirely (`DEC-011`, `DEC-033`) and makes this parameter meaningless rather than refused.
- **Blast radius:** `tt/model_config.py`, `tt/attention/weights.py`, `G-WEIGHTS`, `G-ATTN`,
  `G-MODEL`.

---

### DEC-048 — The weight-cache root is `$TT_CACHE_PATH` and **refuses** to fall back to the checkpoint directory
- **Phase / module:** P6.2 / `tt/model_config.py::weight_cache_path`
- **Date (UTC):** 2026-09-04
- **Trigger:** the template defaults the cache root to the checkpoint directory —
  `models/demos/gpt_oss_d_p/tt/model_config.py:160` is `Path(cache_dir) if cache_dir else
  Path(self.model_path)` — and on this box `$HF_MODEL` already contains `ttnn_cache/` and `P150/`
  from an unrelated package (`07_RISKS.md` R-003).
- **Question:** what happens when `TT_CACHE_PATH` is unset — silently write tilized weights into the
  checkpoint directory, or refuse?
- **Options considered:**
  1. **The template's fallback.** Writes ~8 GB of tilized, mesh-shape-specific tensors into a
     read-mostly checkpoint directory that already holds two foreign caches, where the next run of
     either package may find them.
  2. **A package-local default** (e.g. `<pkg>/.cache`). Convenient, and it puts build artefacts
     inside the source tree, which `G-CLEAN` would then have to exempt.
  3. **Refuse**, with a message naming the variable and the reason.
- **Choice:** option 3. The path is `<$TT_CACHE_PATH>/tensor_cache_<dtype>_<rows>x<cols>`, and
  `cache_root=` overrides it for tests (`tmp_path`).
- **Why:** a tilized tensor is already sharded **and** already cast, so a cache is only valid for
  one (dtype, mesh shape) pair; the recipe's symptom for a stale hit is "one layer runs on garbage"
  three phases later (`BRINGUP_RECIPE.md:939`, Appendix B). Both keys are therefore in the
  directory name, and the root is required rather than guessed — the failure mode of guessing is
  silent and cross-package, which is the worst combination.
- **Evidence:** `G-WEIGHTS`'s `test_weight_cache_path_refuses_the_checkpoint_dir` (raises with
  `TT_CACHE_PATH` unset) and `test_cache_written_at_another_dtype_is_not_reused` (a bf16 build wrote
  12 files into `tensor_cache_bf16_1x1` and **0** into `tensor_cache_bfp8_1x1`; ttnn additionally
  suffixes every file with `_dtype_<DT>_layout_<L>.tensorbin`, so both defences hold).
- **Confidence:** high.
- **Falsifier:** a runner or an engine that cannot set an environment variable and needs the
  fallback. P10's manifest can set `TT_CACHE_PATH` (the `TT_` prefix is one `tt-run` forwards —
  Appendix B), so this is not that case.
- **Revisit if:** P8 finds the mesh shape alone insufficient (e.g. two different TP values on one
  shape), in which case TP joins the path.
- **Blast radius:** `tt/model_config.py`, every module's `tensor_cache_path`, `G-WEIGHTS` and its
  P8 TP=8 extension, `07_RISKS.md` R-003.

---

### DEC-049 — The final norm always runs; `skip_lm_head` skips only the head
- **Phase / module:** P6.3 / `tt/model.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the template returns **pre**-norm hidden states when `skip_lm_head=True`
  (`models/demos/gpt_oss_d_p/tt/model.py:236` returns before the norm at `:240`), and prefill's
  product is the KV cache, so nothing there needed the norm.
- **Question:** what does `prefill_forward(skip_lm_head=True)` return — the residual stream as the
  last layer left it, or the final-normed stream?
- **Options considered:**
  1. **Pre-norm** (the template). `G-MODEL` would then have to re-implement RMSNorm on the host to
     compare against anything HF produces, or score against a quantity no reference exposes.
  2. **Post-norm.** Matches `LlamaModel.last_hidden_state` exactly
     (`.../transformers/models/llama/modeling_llama.py:421` is `hidden_states = self.norm(...)`
     immediately before it is returned), so the gate compares two tensors that mean the same thing.
- **Choice:** option 2. The norm is applied unconditionally; `skip_lm_head` controls the head only.
- **Why:** the gate has to compare against *something a reference produces*, and the only
  hidden-state tensor HF exposes is post-norm. It is also the cheaper end of the trade: one
  `ttnn.rms_norm` over `[1,1,S,4096]` per forward, against a host-side reimplementation of the norm
  in every test that wants hidden states. **And the measurement says the post-norm stream is the
  harder test, not the easier one:** at full depth the last layer's pre-norm output scores
  0.9995853 while the post-norm output scores 0.9984849 — the norm divides out the massive-activation
  channels that dominate the pre-norm correlation and exposes the rest, so gating on pre-norm would
  have flattered the model by ~2.7x in error terms.
- **Evidence:** `.../modeling_llama.py:421` and `:484` (`LlamaForCausalLM` consumes
  `outputs.last_hidden_state` and does not re-expose it); the full-depth curve in
  `raw/G-MODEL_per_layer_pcc.json` (`per_layer_pcc["31"]` vs `final_post_norm_pcc`).
- **Confidence:** high.
- **Falsifier:** a P7/P10 caller that needs the pre-norm stream (a KV-migration consumer would not —
  the cache holds post-RoPE K and raw V, neither of which passes through the final norm).
- **Revisit if:** the final norm ever becomes measurably expensive at long context, in which case it
  becomes an argument rather than an unconditional step.
- **Blast radius:** `tt/model.py`, `G-MODEL`, P7's runtime (which calls `prefill_forward` per chunk
  and discards the hidden states).

---

### DEC-050 — Two per-layer seams: `on_layer_complete(idx)` for P10, `on_layer_output(idx, hidden)` for the bring-up
- **Phase / module:** P6.3 / `tt/model.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the template has one seam, `on_layer_complete(layer_idx)`
  (`models/demos/gpt_oss_d_p/tt/model.py:195`, called at `:210`), and
  `bringup_log/03_OUTLINE.md` §2.11 pins it as "the seam P10's per-layer KV migration hooks attach
  to". `G-MODEL`'s per-layer PCC curve needs the **tensor**, which that signature does not carry.
- **Question:** widen the existing seam to `(layer_idx, hidden_states)`, or add a second one?
- **Options considered:**
  1. **Widen it.** One seam, but P10's migration ack path then receives a live device tensor it has
     no business holding, and every P10 hook has to accept and ignore it.
  2. **Add `on_layer_output(layer_idx, hidden_states)`.** Two callbacks, each with one job: an ack
     seam that carries an index, and a numerical-probe seam that carries the tensor.
  3. **No seam; re-run the model once per layer depth.** 32 forwards to draw one curve, and each
     rebuild would perturb nothing but the wall clock. Rejected on cost, not on correctness.
- **Choice:** option 2.
- **Why:** the two callers want different things at different lifetimes. P10's hook fires to
  acknowledge that a layer's KV is migratable and must stay cheap; the curve probe reads a
  4096-wide activation to the host, which is the opposite of cheap and must not end up on the
  serving path by accident. Keeping them separate means P10 cannot inherit a host readback.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/model.py:195` and `:210` (the template's seam and its
  call site); the curve itself, `raw/G-MODEL_per_layer_pcc.json`, 32 entries from one forward.
- **Confidence:** high.
- **Falsifier:** P10 needing the tensor after all (e.g. to checksum a layer's output before
  migrating its KV), which would make one widened seam the right shape.
- **Revisit if:** P7 or P10 wants a third per-layer hook — three callbacks is a sign the seam should
  become an object.
- **Blast radius:** `tt/model.py`'s `prefill_forward` signature, `G-MODEL`, P10's adapter.

---

### DEC-051 — `G-LAYER`'s error-ratio budget: attribute the fused kernel **through the layer**, and assert both
- **Phase / module:** P6.1 / `tests/unit/test_decoder_layer_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `07_RISKS.md` R-015 hands P6 the same arithmetic that made `G-ATTN` a
  `PASS-WITH-DEVIATION` (`DEC-042`): Appendix A sets `G-LAYER` at "PCC >= 0.999, <= 8x floor"
  (`BRINGUP_RECIPE.md:1877`, stated at `:1372`), and the layer contains the fused SDPA kernel that
  recipe §2.3 measures at 71x its own floor and that accounted for the whole of `G-ATTN`'s block
  gap. The gate's assertion had to be written **before** the number existed.
- **Question:** how is the fused kernel's fixed slack attributed at layer level — and is
  §2.3.1's subtraction even valid there, given the residual add attenuates a perturbation of the
  attention branch by `||y||/||s||` (`BRINGUP_RECIPE.md:1388-1389` measures 1.12x-1.73x)?
- **Options considered:**
  1. **Copy `DEC-042` verbatim**: measure SDPA standalone, subtract its excess from the layer's
     total in the layer's output space. Cheap, and **wrong at layer level** — the kernel's error
     reaches the layer output attenuated by the residual add and then reshaped by the MLP, so
     subtracting the raw excess over-corrects by an unknown factor.
  2. **Model the attenuation** with the recipe's `||y||/||s||` and scale the excess. Introduces a
     modelled constant into a gate whose whole point is measurement.
  3. **Substitute the device's real SDPA output into the floor layer** and let the substitution
     propagate through `o_proj`, the residual add, the second norm and the MLP. The resulting
     `predicted` PCC contains the kernel's real error, attenuated exactly as the arithmetic
     attenuates it, with nothing modelled. Then
     `residual = ((1 - measured) - ((1 - predicted) - (1 - floor))) / (1 - floor)`.
- **Choice:** option 3, asserted at <= 8x at **both** dtypes, plus the raw ratio asserted where
  `DEC-042` says a raw ratio is meaningful (bf8_b), and the raw ratio recorded at both.
- **Why:** it is §2.3.1's instruction — "measure the fused kernel standalone, subtract its excess
  and the floor error from the block's total, and require the remainder ... to sit near 1x"
  (`BRINGUP_RECIPE.md:452`) — with the one term §2.3.1 could not have known about at block level
  (the residual attenuation) handled by construction rather than by a correction factor.
- **Measured, and the finding is that the wall does not bite here:**

  | dtype | S | layer PCC | floor | raw ratio | kernel excess / floor err | attributed residual |
  |---|---|---|---|---|---|---|
  | bf8_b | 128 / 512 / 2048 | 0.9997665 / 0.9998273 / 0.9998736 | 0.9998709 / 0.9998953 / 0.9999138 | **1.81 / 1.65 / 1.47x** | 0.68 / 0.51 / 0.31x | **1.13 / 1.14 / 1.15x** |
  | bf16 | 128 / 512 / 2048 | 0.9998774 / 0.9999172 / 0.9999480 | 0.9999826 / 0.9999859 / 0.9999885 | **7.05 / 5.86 / 4.51x** | 5.04 / 3.83 / 2.37x | **2.01 / 2.03 / 2.14x** |

  So the **raw 8x holds at both dtypes at layer level** — 7.05x is the worst case — where at block
  level it reached 12.32x (`DEC-042`). The reason is arithmetic and worth stating: a layer adds two
  norms and three more bf8_b/bf16 projections to the floor's error budget while the fused kernel's
  absolute slack is unchanged, so the kernel's *share* of the total falls. `G-LAYER` is therefore a
  plain `PASS`, not a `PASS-WITH-DEVIATION`, and `RAW_BLOCK_BUDGET_APPLIES[bf16] = False` — declared
  before the measurement — turns out to have been unnecessary caution rather than a needed escape.
  It is left in place and *not* flipped to `True`: changing a threshold after seeing the number it
  gates is the error `BRINGUP_RECIPE.md:1922` names in both directions, and the residual assertion
  it sits beside is the tighter of the two anyway (2.14x against a budget of 8x).
- **Evidence:** `raw/G-LAYER_20260904T113153Z.log`, all six block cases with their attribution lines. The
  additive model's own check: `predicted` vs `measured` agree to 5-6 decimals in every case
  (bf8_b S=512: predicted 0.9998418, measured 0.9998273).
- **Confidence:** high.
- **Falsifier:** the attributed residual exceeding ~3x while the kernel's excess is unchanged —
  then the additive attribution is wrong and the gap is not the kernel's. Equally: if the raw bf16
  ratio ever exceeds 8x at layer level, this table is the record of what changed.
- **Revisit if:** the SDPA kernel changes, or P8's ring SDPA replaces it on the SP path (it has its
  own slack, measured at 7.98x in the recipe's run).
- **Blast radius:** `tests/unit/test_decoder_layer_vs_ref.py`, `06_GATES.md`'s `G-LAYER` verdict,
  `07_RISKS.md` R-015's P6 half.

---

### DEC-052 — Read the HF oracle through forward hooks, never through `output_hidden_states`
- **Phase / module:** P6.3 / `tests/unit/test_model_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the first `G-MODEL` run measured a hidden-state PCC of **0.9916270** at 2 layers
  while the last-position **logits** PCC on the same forward was **0.9996040**. A model whose
  hidden states are 20x worse than the logits computed from them is arithmetically impossible, so
  one of the two comparisons was measuring the wrong thing.
- **Question:** what exactly is element `i` of `transformers` 5.12.1's `output_hidden_states` tuple?
- **What it actually is, measured:** for an `n`-layer model the tuple is
  `(embeddings, L0_out, ..., L[n-2]_out, POST-FINAL-NORM)` — length `n+1`, and its **last** element
  is `model.norm`'s output, *not* the last layer's. Verified by hooking every module: for `n = 2`,
  `hidden_states[0] == embed_tokens` output, `hidden_states[1] == layers[0]` output, and
  `hidden_states[2] == norm` output, while `layers[1]`'s output appears **nowhere** in the tuple.
  `LlamaForCausalLM`'s `CausalLMOutputWithPast` also exposes **no** `last_hidden_state`
  (`LlamaModel.forward` builds one at
  `python_env/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:421` and
  `LlamaForCausalLM` consumes it at `:484` without re-exposing it), so there is nothing to
  cross-check the tuple against from the causal-LM head.
- **The bug it caused, so the next reader recognises it:** taking `hidden_states[-1]` as the last
  layer's pre-norm output and applying `model.norm` to it computes the norm **twice**. RMSNorm is
  nearly idempotent on an already-normalised tensor — it rescales by roughly the gain — so the
  result is *plausible*: PCC 0.9916, not garbage. The device was correct throughout; on the real
  reference the same layer scores 0.9999551. Diagnosing it took a per-channel comparison that
  showed the "device error" was a uniform ~8x on Llama's massive-activation channels (788, 1384,
  4062 at the BOS positions), i.e. a missing normalisation, not a kernel fault.
- **Options considered:**
  1. **Use `hidden_states[:-1]` for the layers and `hidden_states[-1]` for the post-norm stream.**
     Correct today, and it silently breaks if a future `transformers` appends the last layer's
     output too — with the same plausible-looking failure.
  2. **Register forward hooks** on `model.model.layers[i]` and on `model.model.norm`. Explicit,
     version-independent, and it yields the last layer's pre-norm output, which the tuple does not
     contain at all.
- **Choice:** option 2, for every HF read in the file — reduced-depth, full-depth, floor and control
  alike. `output_hidden_states` is not passed anywhere.
- **Why:** the gate compares tensors across two implementations, and the one thing it cannot afford
  is ambiguity about *which* tensor. A hook names the module it came from.
- **Evidence:** the measurement above; `.../modeling_llama.py:421` and `:484`. The corrected gate
  measures L2/s128 hidden PCC **0.9997314** against a floor of 0.9998795 (2.23x).
- **Confidence:** high — it is a direct measurement of the installed version.
- **Falsifier:** a `transformers` release where the hook-captured layer output and
  `hidden_states[i+1]` disagree; the hooks would still be right.
- **Revisit if:** never for correctness; only if hooking becomes impossible (a compiled/graph HF
  path), in which case run `model.model` directly and norm its `last_hidden_state` yourself.
- **Blast radius:** `tests/unit/test_model_vs_ref.py` (every reference read), `G-MODEL`'s numbers,
  and P7's golden-KV generator, which will read the same oracle.

---

### DEC-054 — Three public surfaces deviate from `03_OUTLINE.md`, each in named places
- **Phase / module:** P6.1-P6.3 / `tt/model_config.py`, `tt/layer.py`, `tt/model.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `bringup_log/03_OUTLINE.md` §2.3, §2.10 and §2.11 pin the constructors and forward
  signatures for these three files, and P3's convention list says "deviating is a `DEC`"
  (`BRINGUP_RECIPE.md:927`). Implementing them produced four deviations.
- **Question / choice, one row each:**

  | Outline | Implemented | Why |
  |---|---|---|
  | `ModelArgs(..., instruct=True, cache_hf=False)` | both dropped | Neither has a consumer in this iteration: `instruct` selects a chat template and nothing here tokenizes through `ModelArgs` (`G-MODEL` uses `AutoTokenizer` directly, `DEC-056`), and `cache_hf` is unread in the template it came from too. Recipe §0 rule 5: no dead parameters. Adding a tokenizer to `ModelArgs` would also make the class import `transformers`, which every dimension-only test currently avoids. |
  | `ModelArgs` has no key-set accessor | `expected_state_dict_keys(n_layers=None)` added | `G-WEIGHTS` must assert "no missing **and** no unused keys" against something; deriving it from the same constants the loader uses is the only version that cannot drift from the loader on a rename. A hand-written list in the test would. |
  | `DecoderLayer(..., dtype=ttnn.bfloat16, max_local_batch_size=1)` | `weight_dtype=bfloat8_b`, `activation_dtype=bfloat16`; `max_local_batch_size` dropped; `attention_config=`/`program_config=` added | One `dtype` argument for two different dtypes is how `DEC-022`'s ladder gets flattened by accident — the sublayers take both separately, so the layer does too. `max_local_batch_size` is unread (it exists in the template for the MoE capacity calculation). The two configs are passed in so `tt/model.py` builds them **once** for all 32 layers; `None` builds them locally, which is what the standalone layer test uses. |
  | `Model.prefill_forward(..., on_layer_complete=None)` | `on_layer_output=` added | `DEC-050`. |
- **Why (the common thread):** each deviation either removes a parameter with no reader or adds one
  a gate needs. None changes a tensor, a shape or a dtype, so no earlier gate's evidence is
  affected.
- **Evidence:** `models/demos/gpt_oss_d_p/tt/layer.py:46`'s signature (the source of the dropped
  arguments, where `max_local_batch_size` and `dtype` are likewise unread by the dense path);
  `bringup_log/03_OUTLINE.md` §2.3, §2.10, §2.11.
- **Confidence:** high.
- **Falsifier:** P7 or P10 calling one of these constructors with `instruct=`, `cache_hf=`,
  `dtype=` or `max_local_batch_size=` because it copied the outline. A `TypeError` at the call site,
  not a silent wrong value.
- **Revisit if:** the package grows a tokenizer-owning surface (P10's request mode may), which is
  where `instruct` belongs.
- **Blast radius:** `bringup_log/03_OUTLINE.md` §2.3/§2.10/§2.11 as written; P7's runtime and P10's
  adapter, which construct these three classes.

---

### DEC-055 — `G-WEIGHTS` proves the per-tensor and cache-only halves on **one** layer, and the key set on all 32
- **Phase / module:** P6.2 / `tests/unit/test_weight_loading.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the recipe asks for "(c) a **sample** of device weights ... bit-exact"
  (`BRINGUP_RECIPE.md:1407`) and a cache-only rebuild producing bit-identical device tensors
  (`:1406`), without saying how much of the model either covers.
- **Question:** how much of the model does `G-WEIGHTS` build with real weights — one layer, or 32?
- **Options considered:**
  1. **All 32 layers, twice** (once from the checkpoint, once from cache). ~290 tensors per build;
     re-proves one loader code path 32 times, at a cost of minutes of host I/O per run and a
     permanent tax on every regression.
  2. **One layer, and every tensor in it** — 12 tensors: embedding, both norm gains, four attention
     projections, three MLP projections, the final norm and the LM head. Plus the **key set** over
     all 32 layers, which is where a per-layer rename would show up and which needs no tensor data
     at all (it reads `model.safetensors.index.json`).
  3. A "sample" in the literal sense — a few tensors. Weaker than option 2 for the same cost.
- **Choice:** option 2. Every weight of a one-layer model, `rtol = atol = 0`, plus 291-key set
  equality across the whole checkpoint.
- **Why:** the loader is written once and parameterised by layer index, so per-layer repetition
  tests the `for` loop, not the loader. What *is* layer-specific is the **key names**, and those are
  covered exhaustively and for free. Checking all twelve tensors rather than a sample also turns
  part (c) into the honest proof of part (a)'s "every expected key is **consumed**": a key that no
  module read could not produce a bit-matching device tensor.
- **Evidence:** `raw/G-WEIGHTS_20260904T113320Z.log` — 291 checkpoint keys, 291 expected, 0 missing, 0 unused;
  12/12 tensors at `max|delta| = 0.000e+00` through the transpose, the Q/K Meta swizzle and the
  bf8_b/bf16 dtype ladder; 12/12 SHA-256-identical after a cache-only rebuild from an empty
  `state_dict`.
- **Confidence:** high on the loader; the **32-layer** cache-only rebuild remains untested at any
  mesh shape, which is what `G-WEIGHTS`'s P8 extension covers (`BRINGUP_RECIPE.md:1411` scopes
  cache-only at TP > 1 there).
- **Falsifier:** a layer-index-dependent loader path — e.g. a per-layer cache subdirectory collision
  — which a 32-layer run would catch and this does not. The cache path is
  `.../model.layers.<i>/...`, so a collision would be a bit-identity failure in the P8 extension.
- **Revisit if:** the loader gains a per-layer branch of any kind.
- **Blast radius:** `G-WEIGHTS`'s runtime and coverage; `08_INTEGRATION`-style coverage claims.

---

### DEC-056 — `G-MODEL`'s input distribution is **real tokenized prompt text**
- **Phase / module:** P6.3 / `tests/unit/test_model_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the recipe requires "**top-1 token agreement = 100%** on the last position"
  (`BRINGUP_RECIPE.md:1421-1422`) but does not say what the input tokens are, and recipe §1.4
  requires every numeric gate to state its input distribution.
- **Question:** what token ids does the gate feed — uniform random over the 128256-way vocab, or
  real text?
- **Options considered:**
  1. **Uniform random ids.** No dependency, and it puts the model far off-distribution: the top-1
     logit gap on noise can be arbitrarily small, so a 100% top-1 requirement becomes a coin flip
     on bf8_b rounding — the gate would be flaky for reasons unrelated to correctness.
  2. **A fixed English prompt through the checkpoint's own tokenizer**, tiled to the sequence
     length. Adds an `AutoTokenizer.from_pretrained($HF_MODEL)` call (the checkpoint ships
     `tokenizer.json`), and makes the top-1 an actual next-token prediction.
  3. **Hard-coded token ids** copied from a tokenizer run. No dependency, but a magic array nobody
     can audit against the text it claims to be.
- **Choice:** option 2, with the prompt a module-level constant so it is visible and fixed.
- **Why:** top-1 agreement is only meaningful where the reference itself is confident. Measured, it
  is: the reference's top-2 logit gap is **3.9214** at 2 layers, and the two implementations agree
  on the argmax at every depth tested. The tiling repeats the tokenizer's BOS, which turns out to be
  a feature — Llama's massive activations live at BOS positions, so the input exercises the highest
  dynamic range in the model at more than one position (that is what made `DEC-052`'s double-norm
  bug visible at all).
- **Evidence:** `raw/G-MODEL_<ts>.log` — top-1 agrees at L2/L4, seq 128/512, and at full depth
  (ref 374 == device 374); the rotated-weight control moves it (`DEC-050`'s sibling assertion).
  Note the top-**5** is not required to agree and does not: at L2/s128 rank 4 differs
  (ref `31240`, device `50294`), which is what a 0.9996 logits PCC looks like and is why the gate is
  on top-1.
- **Confidence:** high.
- **Falsifier:** a top-1 disagreement traceable to the prompt rather than the model — visible as a
  reference top-2 gap near zero, which is logged every run.
- **Revisit if:** the gate ever needs to run on a weightless box (it cannot — it is
  `requires_hf_reference` by construction) or a tokenizer-free machine.
- **Blast radius:** `G-MODEL`'s inputs; P7's golden-KV scripts, which should use the same prompt
  so the two are comparable.

---

### DEC-057 — `G-LAYER`'s norm-swap control also runs on the **real** layer-0 gains
- **Phase / module:** P6.1 / `tests/unit/test_decoder_layer_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the recipe's negative control for `G-LAYER` is "swap the two norm gains (measured
  **0.9471**)" (`BRINGUP_RECIPE.md:1373`). Run on this gate's random weights it measures
  **0.99864** (bf8_b) / **0.99873** (bf16) — below the 0.999 threshold, so it discriminates, but by
  1.4e-3 and nowhere near the recipe's figure.
- **Question:** is the control wrong, is the recipe's number wrong, or is the *input* wrong?
- **The cause, and it is the input.** `random_layer_weights` draws each norm gain as
  `1 + randn * 0.02` (`tests/unit/test_reference_model.py`), so the two gains are within ~2% of each
  other **and** of the identity vector. Swapping two nearly-identical vectors is a nearly-zero
  perturbation; no correct implementation could fail it by much. The recipe's 0.9471 is only
  reachable with gains that actually differ.
- **Options considered:**
  1. **Keep the random-weight control only.** It technically satisfies "the same assertion must
     reject it", with a margin thin enough that a small precision change on either side could flip
     it — a control that can pass by accident.
  2. **Make the random gains more distinct** (e.g. `randn` without the `1 +`). Strengthens the
     control and makes the gate's *positive* arm run on gains no Llama layer has, changing every
     floor in the file.
  3. **Add a second arm on the real layer-0 gains**, keeping the random-weight arm as the
     checkpoint-free one. Both numbers recorded.
- **Choice:** option 3, and the real-gain arm asserts < 0.99 rather than < 0.999.
- **Why:** the two arms answer different questions. The random arm proves the *assertion* is wired
  to the norms at all, on a box with no checkpoint. The real arm is the one that reproduces the
  recipe's finding, because the real `input_layernorm` and `post_attention_layernorm` gains are not
  interchangeable vectors, and it is the arm whose margin makes the control trustworthy.
- **Evidence:** `raw/G-LAYER_20260904T113153Z.log` — the real-gain arm logs both gains' norms and their cosine
  similarity alongside the correct and swapped PCCs.
- **Confidence:** high.
- **Falsifier:** the real-gain arm also landing near 0.999, which would mean the two real gains are
  interchangeable and the control cannot be strengthened this way.
- **Revisit if:** P8's TP=8 layer gate reuses this control (it should, and at TP=8 the gains are
  replicated so the numbers should be unchanged).
- **Blast radius:** `tests/unit/test_decoder_layer_vs_ref.py`'s controls, `G-LAYER`'s block.

---

### DEC-058 — Supersedes `DEC-057`: what makes a residual-block control weak is the input **scale**, not the gains
- **Phase / module:** P6.1 / `tests/unit/test_decoder_layer_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `DEC-057` diagnosed the weak norm-swap control as "the random gains are all ~1, so
  swapping them is a nearly-zero perturbation" and added an arm using the real layer-0 gains. That
  arm was **measured and the diagnosis was wrong**: with real gains (norms 8.994 / 8.595, cosine
  **0.494** — genuinely different vectors) the swap moved the PCC only from 0.9999969 to
  **0.99993**, i.e. *further* from collapsing than the random-gain arm.
- **Question:** why can a wrong norm gain barely move a decoder layer's output, and what input makes
  the control discriminate?
- **The measurement that settles it.** Three arms, same code, same swap:

  | input | weights | max&#124;x&#124; | max&#124;attn_out&#124; | attenuation | swapped-gain PCC |
  |---|---|---|---|---|---|
  | `randn` | random `*0.02`, random gains | ~4 | ~0.5 | small | 0.99864 |
  | `randn` | **real** layer-0, all nine tensors | 5.16 | **0.08** | ~65x | **0.99993** |
  | **real `embed_tokens` rows** | **real** layer-0, all nine tensors | 0.0596 (RMS 0.0106) | 0.3948 | **3.40x** | **0.66830** |

  The mechanism is `BRINGUP_RECIPE.md:1388-1389`'s own: for `y = r + s` a perturbation of `s` is
  attenuated in `y` by `||y||/||s||`. A norm **removes its input's scale**, so a sublayer's output
  magnitude is nearly independent of the input's — while the residual's magnitude *is* the input's.
  Feed a standard-normal `x` (~100x larger than what layer 0 receives, since `embed_tokens` rows
  have an RMS around 0.0106) and the residual drowns both sublayers; feed the real embedding rows
  and the layer amplifies 0.06 to 2.02, the sublayers dominate, and a wrong gain is no longer
  absorbed. The recipe's remark that the masking is "real but small" (1.12x-1.73x) holds for the
  inputs *it* measured; at a 100x input-scale mismatch the same mechanism gives ~65x and swallows
  the control whole.
- **Options considered:**
  1. Keep `DEC-057`'s real-**gain** arm. It passes only because `< 0.99` was asserted against a
     number that measured 0.99993 — i.e. it does not pass at all, and it was never a control.
  2. Rescale a standard-normal input to the embedding's RMS. Gets the magnitude right and the
     *structure* wrong: real embedding rows carry the massive-activation channels (788, 1384, 4062)
     that dominate every downstream norm, and a Gaussian of the same RMS does not.
  3. Drive the arm with **real layer-0 weights and real `embed_tokens` rows** for a fixed token
     sequence — the actual layer-0 input.
- **Choice:** option 3, asserting the gate's own `PCC >= 0.999` must reject the swap.
- **Why:** it is the only arm in which the layer is doing what it does in the model, and it is
  therefore the only arm whose *failure* would mean something. It collapses to **0.66830**, past the
  recipe's quoted 0.9471, with a 0.33 margin instead of 1.4e-3.
- **Evidence:** `raw/G-LAYER_20260904T113153Z.log`, the three arms above. The correct arm on real weights and a
  real input measures **0.9998649** against a floor of **0.9999647** (**3.82x**) — the only
  real-weight, real-input layer number in this gate, and the one comparable with recipe §2.1's
  table, which was also measured on real weights.
- **Confidence:** high. The three-row table is a controlled experiment: one variable changes per row.
- **Falsifier:** an arm with a real-scale input where the swap does *not* collapse. That would mean
  the norms are not wired to the sublayers the way the code says.
- **Revisit if:** any later gate builds a negative control on a residual block — P8's TP=8 layer
  gate and P7's chunked gate both would. **The general rule this produces:** a control on `r + s`
  must be driven at the scale the model actually presents to that block, or the residual absorbs it.
- **Blast radius:** `tests/unit/test_decoder_layer_vs_ref.py`; `G-LAYER`'s control; the design of
  every later residual-block control. Supersedes `DEC-057`.

---

### DEC-053 — `G-MODEL`'s verdict follows the phase text, not Appendix A's compressed row
- **Phase / module:** P6.3 / `tt/model.py` (written by the orchestrator after the P6 session was
  stopped mid-run; the analysis and all numbers are the session's, recorded in `R-020`/`R-021`/`R-022`)
- **Date (UTC):** 2026-09-04
- **Trigger:** at 32 layers / seq 512 / bf8_b weights the post-final-norm hidden PCC is **0.9984849**,
  which is *below* the 0.999 the recipe states — but the recipe states it in two places that disagree
  about the depth it applies at.
- **Question:** does the absolute PCC threshold gate the full-depth run, or only the reduced-depth runs?
- **The conflict:** the phase text (`BRINGUP_RECIPE.md:1420-1425`) attaches `PCC >= 0.999, <= 8x floor,
  100% top-1` to the **reduced** layer counts (2, then 4) and gates the full 32-layer run on the
  **per-layer step (<= 4x from L3)**. Appendix A's single row compressed all of it into one
  unqualified line, which reads as though the absolute threshold also applies at depth 32. Literally,
  Appendix A makes this a `FAIL` that stops the bring-up; the phase text makes it a `PASS`.
- **Choice:** follow the **phase text**. Assert the step and top-1 as stated at full depth, and record
  the absolute number against a *measured full-depth floor* so it has a reference instead of a bare
  comparison to a reduced-depth threshold.
- **Why:** the phase text is the specific instruction and Appendix A is a summary of it; a summary that
  loses a qualifier does not create a requirement. And the evidence says the model is fine: the
  per-layer step never exceeds **1.27x** against a 4x budget, the curve is smooth with no step
  anywhere, **top-1 agrees with HF**, and once the floor is corrected for the bf16 RoPE tables the
  device actually holds (`R-021`) the ratio is **1.53x** — inside the 1.0-2.4x band every other gate
  in this run occupies. The 2.79x that prompted this was a floor defect, not a model defect.
- **Evidence:** `R-020` (the textual conflict, with both line ranges), `R-021` (the floor correction:
  `1 - floor` 5.4300e-04 with fp32 RoPE tables vs 9.9160e-04 with bf16, the omitted term being 45% of
  the correct floor error, and a staged chain reproducing the gate's floor exactly at 0.9994570),
  `R-022` (why §2.3.1's attribution over-subtracts at this depth, residual 0.63x).
- **Confidence:** high on the verdict, medium on the threshold *design* — hence the recipe fix below.
- **Falsifier:** a full-depth run whose per-layer step exceeds 4x, or whose top-1 disagrees, or whose
  corrected ratio leaves the band. None of those hold.
- **Revisit if:** the absolute number is ever needed as a release criterion, in which case it must be
  re-derived at the target depth and sequence length rather than inherited.
- **Blast radius:** `G-MODEL` only. **Fixed upstream in the kit:** Appendix A's `G-MODEL` row is now
  scoped by depth, and new §2.2.4 states the general rule — an absolute PCC threshold without a stated
  depth silently becomes a measure of depth rather than of correctness.

---

### DEC-059 — The golden KV trace is stored at **fp32**, against both templates' `bfloat16`
- **Phase / module:** P7 / `scripts/generate_golden_kv_cache.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** writing the generator. Both in-repo generators default their stored dtype to bf16
  (`models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py:111`,
  `models/demos/minimax_m3/scripts/generate_golden_kv_cache.py:143`), and gpt-oss additionally
  *computes* in bf16.
- **Question:** store the golden at the device's cache dtype (bf16/bf8_b) as the templates do, or at
  fp32?
- **Options considered:**
  1. bf16, as both templates default — matches the device cache byte-for-byte, halves the file size.
  2. **fp32 throughout** — weights cast to fp32 on load, all math fp32, K/V saved fp32.
- **Choice:** option 2, and the generator has no `--dtype` flag at all so there is no way to write a
  bf16 golden by accident.
- **Why:** recipe §2.1(a) is about exactly this. A reference held at the device's own storage dtype
  *shares the device's rounding*, so every PCC scored against it is flattered and the noise floor
  becomes unmeasurable (there is nothing left between the reference and the floor). The recipe states
  it directly: `BRINGUP_RECIPE.md:1543`, "Store the golden at **fp32**, not the template's bf16: it is
  the reference".
- **Evidence:** measured at `G-CHUNK`. The layer-0 storage floor — the golden quantised to bf8_b —
  is **0.9999716** on K and **0.9999628** on V. A bf16 golden would have sat *between* the fp32
  reference and that floor, so the recorded ratio would have been meaningless rather than 1.10x /
  1.02x. Cost: 128 MB for 32 layers x 512 tokens, versus 64 MB at bf16.
- **Confidence:** high.
- **Falsifier:** a long-context trace whose fp32 size is prohibitive. At the deployment 128k context
  the trace would be 32 GB, which is when this needs revisiting — not before.
- **Revisit if:** a 128k golden is needed; then store fp32 for the gated prefix and bf16 beyond it,
  and record which is which in `metadata.json`.
- **Blast radius:** `G-GOLDEN`, `G-CHUNK`, and P8's `G-MESH-KV`/`G-KV-TP8`, which score against the
  same trace.

---

### DEC-060 — `verify_golden_kv.py` follows the `G-GOLDEN` gate text, not P7 step 2
- **Phase / module:** P7 / `scripts/verify_golden_kv.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the recipe describes this one file two incompatible ways.
- **The conflict, quoted:** `BRINGUP_RECIPE.md:1548-1549` — "`scripts/verify_golden_kv.py` — compare
  a device KV read-back against the golden, per layer, reporting min/mean PCC per layer for K and V".
  `BRINGUP_RECIPE.md:1588-1590` — "**Gate `G-GOLDEN`:** `verify_golden_kv.py` runs clean over all 32
  layers and prints a per-layer table ... **It imports no ttnn** — the device-vs-golden scoring lives
  in `G-CHUNK`." A file that compares a *device* read-back must import ttnn. The two cannot both hold.
- **Question:** is this file a device-vs-golden scorer or a host-only structural checker?
- **Choice:** the **gate text**. `verify_golden_kv.py` imports no ttnn and scores nothing against the
  device; the device-vs-golden PCC is `tests/unit/test_attention_chunked_vs_ref.py`'s (`G-CHUNK`).
- **Why:** three reasons, in order of weight. (1) The gate is the thing with a verdict, and Appendix A
  agrees with it (`BRINGUP_RECIPE.md:1976` gives `G-GOLDEN` device "host (imports no ttnn)"). (2) Both
  in-repo templates are host-only structural checkers (`models/demos/minimax_m3/scripts/verify_golden_kv.py:26`,
  `models/demos/gpt_oss_d_p/scripts/verify_golden_kv.py:111`), so following the gate is also following
  the reuse rule. (3) A device scorer here would duplicate `G-CHUNK` and give two gates two ways to
  disagree about the same number.
- **Evidence:** the two recipe passages above; the two template files; `bringup_log/03_OUTLINE.md`
  §2.14, which already contracted this file as "Structural check over all 32 layers ... **Imports no
  ttnn**" and was gated at `G-OUTLINE`.
- **Confidence:** high.
- **Falsifier:** if P10's producer read-back turns out to need a host-side PCC helper that does not
  belong in a test file, this script would be the place for it — but it would still not import ttnn,
  because the producer's reader is device-less by design.
- **Revisit if:** P10 needs a device-less golden-vs-read-back comparator; add it as a second entry
  point rather than changing this one's contract.
- **Blast radius:** `G-GOLDEN` only. **Kit fix owed:** `R-025`.

---

### DEC-061 — `CHUNK_SIZE = 8192` and `MAX_SEQ_LEN = 131072` for deployment; `128` for `G-CHUNK`
- **Phase / module:** P7 / `tt/tt_prefill_runtime.py`, `tests/unit/test_attention_chunked_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `DEC-004` deferred both numbers to P7 and `07_RISKS.md` `R-004` carries them. Nothing
  downstream can be built without them: the indexed RoPE table is built once per chunk size and the
  cache capacity fixes every allocation.
- **Question:** which `(chunk_size, max_seq_len)` pair does the deployment use, and which does the
  single-card gate use?
- **Constraints (both from `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md`'s shared
  setup, via `bringup_log/00_MODEL_CARD.md` §4):** `CHUNK_SIZE % (SP * 32) == 0` and
  `MAX_SEQ_LEN % CHUNK_SIZE == 0`, with SP = 4 on the target `(4,8)` mesh.
- **Choice and the arithmetic:**
  - **`MAX_SEQ_LEN = 131072`** — read verbatim from `config.json:max_position_embeddings`
    (`bringup_log/00_MODEL_CARD.md` §2), so it is the model's own context rather than a number of ours.
  - **`CHUNK_SIZE = 8192`** — `8192 % (4 * 32) = 8192 % 128 = 0`; `131072 / 8192 = 16` chunks exactly.
    It is also the in-repo GQA template's default (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:63`).
  - **`G-CHUNK` runs `CHUNK_SIZE = 128`** at `max_seq_len = 512`: `128 % 128 = 0` (so it is legal at
    the deployment SP too, not just at the gate's `sp = 1`), and `512 / 128 = 4` chunks, giving write
    offsets `{0, 128, 256, 384}` — four distinct `kv_actual` values rather than one.
- **Why not the other template's 5120:** `models/demos/minimax_m3/tt/tt_prefill_runtime.py:49` uses
  5120, which suits its 8-row SP axis. It **fails the second constraint here**: `131072 % 5120 = 3072`.
  Measured, not assumed.
- **Why not a larger gate chunk:** the gate needs several chunks to exercise the advancing offsets,
  and the whole 32-layer model plus two KV producers has to fit on one card. 4 x 128 does that in 67 s.
- **Evidence:** `G-RUNTIME`'s `test_deployment_chunk_geometry` asserts every step of the arithmetic
  above; `G-CHUNK` ran the 4-chunk geometry.
- **Confidence:** high on the constraints, medium on 8192 as the *performance* choice — this iteration
  is functional-first and no timing was measured.
- **Falsifier:** a chunk size at which the ring SDPA's short-Q/long-K requirement fails, or at which
  L1 cannot hold the per-chip chunk. P8 measures both.
- **Revisit if:** P8's `G-CHUNK-ATTN` or P10's `G-REQUEST` shows 8192 is wrong for the ring path;
  `additional_chunk_sizes` makes a second size a config change rather than a code change.
- **Blast radius:** `tt/tt_prefill_runtime.py`, `G-CHUNK`, `G-RUNTIME`, and every P8/P10 gate that
  allocates a cache. **Closes `R-004`.**

---

### DEC-062 — The runtime owns no KV cache: no `owns_kv_cache` branch, and `compile(kv_caches)` is required
- **Phase / module:** P7 / `tt/tt_prefill_runtime.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the template carries a cache-ownership flag
  (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:12`, `owns_kv_cache=True` for its standalone
  galaxy harness) and every cache-touching method defaults to `self.kv_cache`.
- **Question:** carry the ownership branch for a future standalone harness, or make the engine the
  only owner?
- **Choice:** engine-only. There is no `self.kv_cache`, `_resolve_kv(None)` **raises**, and
  `compile(kv_caches)` takes the cache as a required positional argument rather than defaulting to
  `None` as `bringup_log/03_OUTLINE.md` §2.12 sketched.
- **Why:** the contract says the engine allocates the cache and passes it into every call that
  touches it (`models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md:101-107`), and recipe P7
  step 3 says "The runtime must not own the KV cache". The failure mode a fallback creates is
  specific and silent: with `self.kv_cache` present, a harness that forgets to pass the engine's
  cache writes the runtime's *own* cache and every later read of the engine's cache returns zeros —
  which reads as a numerical bug in attention, not as a plumbing bug.
- **Evidence:** `G-RUNTIME`'s `test_prefill_chunk_refuses_a_missing_or_foreign_cache` (both the
  `None` and the wrong-type refusals) and `test_prefill_chunk_accepts_a_one_element_sequence_of_caches`
  (the template's `kv_caches[0]` form still works, so a harness written to the template is not broken
  by this).
- **Confidence:** high.
- **Falsifier:** a P8 harness that genuinely needs to own a cache. It can allocate one and pass it —
  that is one line, and it keeps the ownership visible at the call site.
- **Revisit if:** never for this reason; if a standalone harness appears, it owns the cache itself.
- **Blast radius:** `tt/tt_prefill_runtime.py`, `G-RUNTIME`, and P10's adapter, which is now the only
  place a cache is allocated.

---

### DEC-063 — The unimplemented engine hooks are **present and raise**, rather than absent
- **Phase / module:** P7 / `tt/tt_prefill_runtime.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the `G-RUNTIME` AST walk found that the engine calls **nine** methods on a runtime
  handle, four of them unguarded on paths this package will not implement until P10:
  `build_kv_chunk_table` (`models/demos/common/prefill/runners/prefill_runner.py:570`, `:644`,
  `:699`), `set_layer_ack_channel` (`:746` is `set_d2h_ack_service`; the ack channel is at `:768`),
  `set_layer_completion_sink` (`:752`) and `set_d2h_ack_service` (`:746`). Two more,
  `kv_migration_stages` and `kv_migration_base_address`, are reached behind `hasattr` (`:613`, `:616`).
- **Question:** omit the unimplemented hooks (so the engine's own guard or its own error fires), or
  define each one and have it raise?
- **Options considered:**
  1. **Omit them.** For the `hasattr`-guarded pair the engine then raises its own clear
     `RuntimeError` (`prefill_runner.py:619-623`) naming the doc's §2. For the unguarded four it is
     an `AttributeError` from inside the engine.
  2. **Define each and raise `NotImplementedError`**, naming the owning phase and the risk id.
- **Choice:** option 2 for all six.
- **Why:** an `AttributeError` from inside the engine names the engine, not the phase that owes the
  work; a `NotImplementedError` from here names P10, the gate that will cover it (`G-KV-TABLE`,
  `G-LOOPBACK`) and `R-024`. The recipe's own rule for this is P10 step 4 — "Anything you do not
  implement ... must **raise**, naming its risk id, rather than silently discarding an argument" —
  and a `build_kv_chunk_table` that returned its `path` argument unchanged would let a migration run
  publish nothing and report success.
- **The cost, stated:** defining `kv_migration_base_address` changes which engine branch runs
  (`hasattr` at `:616` becomes true), so the engine's own diagnostic at `:619-623` is no longer
  reachable. That is a real loss and it is why this is a `DEC` rather than an obvious call; the
  replacement message is strictly more specific, so the trade is worth it.
- **Evidence:** `G-RUNTIME`'s `test_unimplemented_engine_hooks_refuse_loudly` (five hooks, each
  matched on its message) and `test_every_raise_in_the_module_is_covered` (25 `raise` statements
  counted in the module).
- **Confidence:** high.
- **Falsifier:** if the engine ever treats a `NotImplementedError` from a hook as recoverable and
  continues, this becomes worse than absence. It does not today — none of the four unguarded call
  sites is inside a `try`.
- **Revisit if:** P10 implements them, at which point the `raise` bodies become the real
  implementations.
- **Blast radius:** `tt/tt_prefill_runtime.py`, `G-RUNTIME`, P10's whole ladder.

---

### DEC-064 — `G-CHUNK`'s layer-0 ratio is asserted against a **complete** floor, with the recipe's storage floor recorded beside it
- **Phase / module:** P7 / `tests/unit/test_attention_chunked_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:1583` names the layer-0 comparison "**≤ 3x** the bf8_b **storage
  floor**", and `07_RISKS.md` `R-021` closes with "P7/P8 must apply the corrected floor definition to
  `G-CHUNK`". The two instructions are not the same floor.
- **Question:** which floor does the 3x budget apply to?
- **The difference:** a *storage* floor quantises only the tensor the cache holds — it is
  `PCC(golden, quantise(golden, bf8_b))` and models the write and nothing else. The device path also
  pays the bf16 embedding output it starts from, the bf16 norm gain, the **bf8_b projection weight**
  and the bf16 RoPE tables. A floor that omits those is incomplete in exactly the way that produced
  `G-MODEL`'s apparent 2.79x (`R-021`), where one omitted term was 45% of the correct floor error.
- **Choice:** compute **both**, log both, and assert the budget against the **complete** floor. The
  complete floor quantises: the producer's input (bf16), the norm gain (bf16), the projection weight
  (bf8_b, quantised in the *transposed, Meta-swizzled* orientation the device stores it in, then
  mapped back — the `_device_valued` helper `G-MODEL` already uses), the RoPE cos/sin (bf16), and the
  stored output (bf8_b, quantised **after** the HF→Meta permutation because bf8_b's shared exponent
  is per 16-element block of the last dim). No internal intermediate is quantised, which is §2.2's
  conservative reading.
- **Measured (layer 0, seq 512, 4 chunks of 128):**

  | quantity | K | V |
  |---|---|---|
  | measured PCC (both producers, identical) | 0.9999617 | 0.9999369 |
  | storage floor (recipe's) | 0.9999716 | 0.9999628 |
  | ratio to the storage floor | **1.35x** | **1.70x** |
  | complete floor | 0.9999651 | 0.9999382 |
  | ratio to the complete floor | **1.10x** | **1.02x** |

- **Why this is worth a decision even though both pass:** they *both* clear 3x here, so no verdict
  turns on it — but the two numbers differ by 23% on K and 67% on V, and the storage floor is the one
  that inflates. Had the projection-weight term been larger (it is the dominant term in `G-MLP`,
  where bf8_b weights cost 7.9e-05 of floor error), the storage floor would have manufactured a
  finding out of a correct implementation. Recording both is what makes the number interpretable
  rather than merely green.
- **Evidence:** `raw/G-CHUNK_20260904T124825Z.log`, `raw/G-CHUNK_per_layer_pcc.json`
  (`layer0_floors`, `layer0_ratios`).
- **Confidence:** high.
- **Falsifier:** a complete-floor ratio below 1.0, which §2.3 defines as a broken floor rather than a
  kernel beating arithmetic. Measured 1.02-1.10x, so the floor is complete but not over-complete.
- **Revisit if:** the cache dtype changes; the floor's term list is dtype-specific.
- **Blast radius:** `G-CHUNK`; P8's `G-KV-TP8` and `G-MESH-KV` should use the same term list.
  **Applies `R-021` to P7 as that risk requires.**

---

### DEC-065 — `G-CHUNK` gets **two** negative controls, one per delta, because the recipe's one cannot move V
- **Phase / module:** P7 / `tests/unit/test_attention_chunked_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:1585-1586` specifies one control — "rope every chunk at
  `kv_actual_global = 0` and the mutual PCC must collapse (measured 0.706 / 0.655)" — and quotes a
  **pair** of numbers, which implies it moves two quantities.
- **The problem:** in this gate's decomposition both producers are fed *the same* hidden states, and
  **V is never rotated**. So freezing the RoPE offset cannot change V by any amount: it is a K-only
  discriminator, and the second of the recipe's two numbers is not reachable by a correct
  implementation of the gate the recipe itself specifies. (The quoted pair is presumably from a
  variant that re-ran the layer stack per chunk, where a wrong K corrupts attention and therefore the
  next layer's V.)
- **Question:** accept a control that exercises delta 1 only, or add one for delta 2?
- **Choice:** both. `freeze_rope_offset` breaks delta 1 (rope every chunk at 0);
  `freeze_write_offset` breaks delta 2 (write every chunk at `kv_actual = 0`, so chunk 3 overwrites
  chunk 0's rows). The delta-1 control asserts K collapses **and** that V is *unchanged*, which turns
  the V invariant into an assertion instead of a footnote.
- **Measured:**

  | control | mutual K | mutual V | verdict |
  |---|---|---|---|
  | delta 1 — rope every chunk at 0 | **0.72466** | 1.00000 (invariant) | discriminates on K |
  | delta 2 — write every chunk at 0 | **0.22048** | **0.04473** | discriminates on both |

  The recipe's 0.706 and this run's 0.72466 agree to within 3%, which is good evidence the delta-1
  control is the same experiment.
- **Why it matters:** a gate covering two deltas with one control leaves the second delta ungated. If
  `kv_actual` were dropped on the floor — the single most likely chunked-prefill bug, and exactly the
  landmine `write_kv_chunk handed a multi-head tensor` sits next to — the delta-1 control would still
  pass.
- **Evidence:** `raw/G-CHUNK_20260904T124825Z.log`, both control lines.
- **Confidence:** high.
- **Falsifier:** a delta-1 control that *does* move V would mean V is being rotated somewhere, which
  is a real bug; the assertion is written that way round on purpose.
- **Revisit if:** P8's `G-CHUNK-ATTN` re-runs the layer stack per chunk, where the recipe's paired
  numbers become reachable and both should be recorded.
- **Blast radius:** `G-CHUNK` only. **Kit fix owed:** `R-027`.

---

### DEC-066 — The golden trace lives outside the repo, addressed only by `$PREFILL_TRACE_DIR`
- **Phase / module:** P7 / `scripts/generate_golden_kv_cache.py`, `tests/unit/test_attention_chunked_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the 32-layer fp32 trace is **128 MB**, and the repo's `pre-commit` rejects files over
  500 KB.
- **Question:** where does the trace live, and how does a gate find it?
- **Choice:** outside the repo — this run used `/home/mstojkovic/prefill_traces/llama31_8b_d_p/s512` —
  addressed by **`$PREFILL_TRACE_DIR`** only. `G-CHUNK` carries a `requires_golden_trace` skip marker
  in the same shape as `requires_hf_reference`, so the suite still runs without a trace.
- **Why:** the recipe is explicit that the variable is the engine's and the package must not invent
  one (`BRINGUP_RECIPE.md:1546-1547`), and the only gitignored directory inside the package
  (`generated/`) is where ttnn writes its own inspector and watcher artifacts — putting a 128 MB
  reference trace there would mix evidence with scratch.
- **The cost, stated:** the trace is **not** reproducible from the repo alone. It is reproducible from
  the repo *plus the checkpoint* by one command, and the generator's own log
  (`raw/G-GOLDEN-GEN_20260904T123642Z.log`) records the command, the shapes and the
  `rtol=atol=0` cross-check — so the evidence chain is intact even though the bytes are not committed.
  `R-026` carries it.
- **Evidence:** `raw/G-GOLDEN_20260904T125230Z.log` (the verifier on the real trace and both negative
  controls); `metadata.json`'s `token_ids`, which is what makes `G-CHUNK` reproducible against a
  regenerated trace.
- **Confidence:** high.
- **Falsifier:** a reviewer who cannot regenerate the trace. The generator needs only `$HF_MODEL` and
  100 s of CPU.
- **Revisit if:** CI needs the gate; then a small (2-layer, 64-token) trace could be committed as a
  structural fixture, which would not be a reference but would keep the plumbing tested.
- **Blast radius:** `G-CHUNK`, `G-GOLDEN`, and P8/P10, which score against the same trace.

---

### DEC-067 — `make_chunk_input` returns a **4D** per-chip tensor, following the outline over the template
- **Phase / module:** P7 / `tt/tt_prefill_runtime.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `bringup_log/03_OUTLINE.md` §2.12 contracted `[1, 1, 1, chunk_size/SP]` per chip; the
  template produces a **3D** per-chip tensor (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:231`
  reshapes to `(sp, 1, s_local)`) and relies on `unsqueeze_to_4D` downstream.
- **Question:** follow the template's 3D shape or the outline's 4D?
- **Choice:** 4D — `torch.tensor(...).reshape(sp, 1, 1, chunk_local)` sharded on the SP axis.
- **Why:** `tt/embedding.py::Embedding.__call__` documents its input as `[1, 1, 1, S]` (or `[1, S]`)
  and `tt/model.py::prepare_inputs_prefill` already reshapes to `[1, 1, 1, seq_len]`, so 4D is the
  package's own convention and the two entry points into the embedding then agree. The `Embedding`
  wrapper handles 3D too, so this is a consistency choice rather than a correctness one — hence a
  `DEC` and not a silent deviation.
- **Evidence:** `tt/embedding.py`'s `__call__` docstring and its `unsqueeze_to_4D` branch;
  `G-RUNTIME`'s `test_doc_required_names_are_present` covers presence, not shape — the shape is
  first exercised on device in P8, which is stated in `G-RUNTIME`'s "what this does not prove".
- **Confidence:** medium. The shape is **not** exercised on device in P7, because the runtime cannot
  be instantiated on one card at all (`DEC-062`'s TP equality).
- **Falsifier:** a `ttnn.embedding` that rejects the 4D input at P8. It accepts `[1,1,1,S]` today via
  `prepare_inputs_prefill`, which is the same call.
- **Revisit if:** P8's first real chunk fails on the input spec.
- **Blast radius:** `tt/tt_prefill_runtime.py`, P8's first served chunk, P10's H2D path.

---

### DEC-068 — `G-RUNTIME` reaches the per-chunk refusals with a mesh stub and an unbuilt runtime, to stay device-free
- **Phase / module:** P7 / `tests/unit/test_prefill_runtime_chunked.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** Appendix A gives `G-RUNTIME` device **"none"** (`BRINGUP_RECIPE.md:1977`), but half its
  refusals live in `prefill_chunk` and `compile`, which are instance methods — and the runtime cannot
  be instantiated on the machine P7 has, because `tp == num_key_value_heads` refuses `(1,1)`
  (`DEC-062`).
- **Question:** move the gate to the mesh (which P7 is told not to do), or reach the methods without a
  device?
- **Options considered:**
  1. Run `G-RUNTIME` on the `(4,8)` mesh. Rejected: `BRINGUP_RECIPE.md:1623` says "do not move P7 to
     a multi-device mesh to make it run", and Appendix A gives this gate no device.
  2. Test only the refusals reachable from the config's `__post_init__`. Rejected: that leaves the
     delta-3 refusal — the one the recipe specifically requires — untested.
  3. **A `_MeshStub` exposing only `.shape`** for the two `__init__` refusals, and
     `object.__new__(TtPrefillRuntime)` with the three attributes the argument checks read
     (`config`, `rope_indexed`, `hf`) for the per-chunk ones.
- **Choice:** option 3, and the runtime's `prefill_chunk` was **reordered** so that every pure-argument
  check runs before the cache is resolved or any op is issued — which is better design independently
  (a bad chunk range should be refused on its own terms, not behind a cache-type error) and is what
  makes the device-free test honest rather than a mock of the real path.
- **Why this is not a mock:** the code under test is the real method on the real class with the real
  messages; only the state it reads is supplied directly. The `_MeshStub` reaches the two `__init__`
  refusals because both run before any `ttnn` object is constructed — asserted by the tests passing
  with no device open.
- **Evidence:** `raw/G-RUNTIME_20260904T125242Z.log` — 37 tests, 8.2 s, no mesh opened.
- **Confidence:** high on the refusals; the *happy path* of `prefill_chunk` is entirely uncovered by
  this gate and that is stated in the gate block.
- **Falsifier:** a refusal that depends on state the stub does not supply would raise `AttributeError`
  instead of its own error, and `expect_error` would report the wrong type. None does.
- **Revisit if:** P8 instantiates the runtime on the `(4,8)` mesh, at which point these refusals should
  be re-run against a real instance as a cheap extra.
- **Blast radius:** `G-RUNTIME` only.

---

### DEC-069 — `G-GOLDEN` stays a **script** gate, so the per-phase regression does not cover it
- **Phase / module:** P7 / `scripts/verify_golden_kv.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `bringup_log/03_OUTLINE.md` §Gate-ownership table gives `G-GOLDEN` the mechanism
  "script, exit code", while Appendix A also requires "a **per-phase regression gate** ... the whole
  package suite, 0 failed, after each phase's additions", which only sees pytest.
- **Question:** add a pytest wrapper so the regression covers the golden checks, or leave it a script?
- **Choice:** leave it a script, as the outline contracted, and rely on `G-CHUNK` to exercise the trace
  from inside the suite (it reads `metadata.json` and all 32 layer files on every run).
- **Why:** the gate's negative controls are *file-system* mutations of a 128 MB trace — zero a layer,
  delete a layer — and a pytest version would either copy 256 MB per run or test a synthetic trace
  that is not the one the other gates use. The script form tests the real artefact.
- **The gap, stated:** a change that breaks `verify_golden_kv.py` will not be caught by the P7
  regression run. It will be caught the next time `G-GOLDEN` is run, which is P8's first act (it
  scores against the same trace).
- **Evidence:** `raw/G-GOLDEN_20260904T125230Z.log` — the verifier exits 0 on the real trace, 1 with
  seven named problems on the zeroed-layer control, and 1 on the deleted-layer control.
- **Confidence:** medium; a wrapper is cheap and a later phase may want one.
- **Revisit if:** P9's cleanliness gate wants every gate in one command.
- **Blast radius:** the P7 regression's coverage, nothing executable.

---

## P8 — Multi-device: TP, SP and the CCL gates

---

### DEC-070 — `G-FABRIC-MATRIX` runs every case in a subprocess with a timeout, and resets the box after a hang
- **Phase / module:** P8 / `tests/fabric_topology_matrix.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** recipe P8 step 2 requires the sweep, and two of its cases do not fail — they **hang**,
  and the hang is not contained: every later collective on the box hangs too, until `tt-smi -r`
  (`BRINGUP_RECIPE.md:1706-1712`).
- **Question:** pytest with a per-test timeout, or a bespoke parent/child harness?
- **Options considered:**
  1. pytest + `pytest-timeout`. The repo already has it (it fired at 300 s on `G-WEIGHTS`'s P8 arm).
     But `pytest-timeout`'s default method cannot reliably interrupt a device call blocked in C++,
     and a hang would poison every *remaining* test in the same session — turning one measurement
     into a lost run, which is the exact failure the recipe describes.
  2. A parent process that runs each case as `python fabric_topology_matrix.py --case <id>` with
     `subprocess.run(timeout=...)`, and calls `tt-smi -r` after any timeout.
- **Choice:** option 2. `CASE_TIMEOUT_S = 240`.
- **Why:** a hang has to become a *recorded measurement*, which means the process that hangs must
  not be the process that records. 240 s is ~6x the slowest healthy case measured (29.5 s), so a
  `hang` verdict is a hang and not a slow machine.
- **The recovery path was validated before the first hazardous case ran**, not after: `tt-smi -r`
  exit 0 in 41.8 s on this box, and 43.1 s when the harness invoked it for real.
- **Evidence:** `raw/G-FABRIC-MATRIX_20260904T142819Z.log` — `overlap_1x2_then_1x8_no_quiesce`
  recorded `hang` at 246.3 s, the box was reset in 43.1 s, and the next case
  (`overlap_1x2_then_1x8_quiesce`) passed bit-exactly 22.4 s later. A poisoned box would have made
  that impossible.
- **Confidence:** high.
- **Falsifier:** if a healthy case ever took more than 240 s the harness would mislabel it a hang.
  Every healthy case measured 18-30 s.
- **Revisit if:** a case is added whose legitimate runtime approaches the timeout.
- **Blast radius:** `G-FABRIC-MATRIX` only.

---

### DEC-071 — Neither mesh-graph descriptor the recipe names is usable on one galaxy
- **Phase / module:** P8 / `tests/fabric_topology_matrix.py`, `tests/test_factory.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:80-83` says "BH Galaxy mesh descriptors live in
  `tt_metal/fabric/mesh_graph_descriptors/` — e.g. `bh_galaxy_sp4_torus_xy_graph_descriptor.textproto`,
  `32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto`. The Ring topology P8 needs the torus
  descriptor". The user's hard scope limit for this session is **one** Blackhole Galaxy.
- **Question:** which descriptor does a single galaxy use?
- **Finding:** **neither of the two named.** Read from the files themselves:
  - `bh_galaxy_sp4_torus_xy_graph_descriptor.textproto:1-11` declares **four** meshes of
    `dims: [32, 4]` with `host_topology { dims: [4, 1] }` — a super-pod of 4 galaxies, 512 devices,
    16 hosts;
  - `32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto:3-10` declares one `dims: [32, 4]` mesh
    with `host_topology { dims: [4, 1] }` — a quad galaxy, 128 devices, 4 hosts.
  Both are multi-galaxy and out of scope. The single-galaxy descriptors are
  `single_bh_galaxy_mesh_graph_descriptor.textproto` (`dims: [8, 4]`, no `dim_types`, i.e. LINE/LINE)
  and `single_bh_galaxy_torus_xy_graph_descriptor.textproto` (`dims: [8, 4]`,
  `dim_types: [RING, RING]`), both `host_topology { dims: [1, 1] }` — 32 devices, one host. The
  in-repo galaxy harness names the second one
  (`models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:26`).
- **Choice:** the harness sets `TT_MESH_GRAPH_DESC_PATH` to `single_bh_galaxy_torus_xy...` for its
  `FABRIC_1D_RING` cases and `single_bh_galaxy_mesh...` for its `FABRIC_1D` cases, always explicitly,
  never by default.
- **Why:** the descriptor and the fabric config are one decision (`DEC-027`), and leaving either to
  a default is how a Ring collective ends up on a route that does not exist.
- **Evidence:** the descriptor files; `raw/G-FABRIC-MATRIX_20260904T142819Z.log`.
- **Confidence:** high — read from the files, not inferred.
- **Blast radius:** every P8 gate's fabric configuration; the recipe's §The machine bullet.

---

### DEC-072 — The SP path needs **two** cores, not one: the ring, and an exact bootstrap
- **Phase / module:** P8 / `tt/attention/dense_sp.py`, `tt/attention/prefill.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** implementing the P5.5 seam. The recipe's P8 step 7 names only the ring path.
- **Question:** is `dense_sp_attention` enough to cover every SP > 1 configuration?
- **Finding:** no. `ttnn.transformer.ring_joint_scaled_dot_product_attention`'s own docstring:
  "Chunked-prefill mode is entered implicitly when `input_tensor_q`'s per-device seq length is
  **less than** `input_tensor_k`'s". A request whose single chunk fills the whole cache
  (`max_seq_len == chunk_global`) makes them equal, so the ring op cannot serve it.
- **Choice:** implement `sp_bootstrap_attention` as well — all-gather Q/K/V on the SP axis, run the
  **caller's own** dense SDPA closure on the full sequence, reduce-scatter, divide by `sp` (the
  reduce-scatter sums `sp` identical copies). Template:
  `models/demos/gpt_oss_d_p/tt/attention/prefill.py:230-252`.
- **Why:** it makes a one-shot SP request exact rather than unsupported, and it gives `G-MESH-KV`
  and `G-CHUNK-ATTN` a genuinely different second core to compare the ring against. Taking the
  `run_sdpa` closure from the caller rather than building a program config here is what guarantees
  the bootstrap runs the same kernel every P5-P7 gate scored.
- **The hazard it introduces, and how it is closed:** Appendix B's last row — "everything passes but
  the numbers look too good | you measured the SP bootstrap because `max_seq_len == chunk_size`".
  `select_attention_core` (`DEC-075`) names the choice in one place and every gate **asserts** which
  core ran.
- **Evidence:** `raw/G-MESH-KV-oneshot_20260904T150307Z.log` (`attention core for chunk 0:
  sp_bootstrap`, min K 0.9987994) vs `raw/G-MESH-KV-chunked512_20260904T150451Z.log.gz`
  (`sp_ring`, min K 0.9967119).
- **Confidence:** high.
- **Falsifier:** if the ring op ever accepted equal Q and K lengths the bootstrap would be dead code.
- **Blast radius:** `tt/attention/prefill.py`, `G-MESH-KV`, `G-CHUNK-ATTN`.

---

### DEC-073 — The ring SDPA's `q_chunk_size` / `k_chunk_size` stay at the template's 128
- **Phase / module:** P8 / `tt/attention/dense_sp.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `sp_ring_program_config` needs SDPA chunk sizes, and the dense path's are seq-len
  dependent (32 below 2048, 256 above).
- **Question:** derive them from the chunk length, or pin them?
- **Choice:** pin `q_chunk_size = k_chunk_size = 128`, the template's value
  (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:196-197`).
- **Why:** the ring op's Q slab is one chunk rather than the whole sequence, so the dense path's
  threshold logic does not transfer; and recipe §2.3 measured that sweeping SDPA chunk sizes over
  {32, 128, 256} moves the fused kernel's PCC by **under 4%**, so this is not a correctness knob in a
  functional-first iteration.
- **Measured, and it is the reason this entry exists rather than being an inherited default:** the op
  accepts `q_chunk_size=128` with a per-device Q of **64** rows — `G-MESH-KV` at
  `PREFILL_CHUNK_SIZE=256` (chunk_local 64) scored min K 0.9967844 / V 0.9866232, essentially equal
  to the chunk_local-128 arm's 0.9967119 / 0.9868228. A pinned 128 therefore does not constrain the
  deployable chunk size, which was the only reason to consider deriving it.
- **Evidence:** `raw/G-MESH-KV-chunked256_20260904T150549Z.log.gz`,
  `raw/G-MESH-KV-chunked512_20260904T150451Z.log.gz`.
- **Confidence:** high for correctness, none claimed for performance.
- **Revisit if:** perf work starts.
- **Blast radius:** `G-SP-RING`, `G-CHUNK-ATTN`, `G-MESH-KV`.

---

### DEC-074 — `dense_sp_attention`'s `q` is the **per-device** chunk, correcting the P5.5 seam's docstring
- **Phase / module:** P8 / `tt/attention/dense_sp.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the P5.5 stub documented `q` as `[1, n_q_local, chunk_global, head_dim]`.
- **Finding:** wrong. The op takes `[b x nh x N/num_devices x dh]` (its own docstring), i.e. the
  **per-device** length `chunk_local`; the global figure travels in `logical_n`. The template's own
  call passes the post-head-split per-device tensor
  (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:208-216`).
- **Choice:** fix the docstring to `chunk_local` and say so here rather than silently.
- **Why:** the seam's whole purpose (P5.5) was that P8 would not have to rediscover the interface; a
  wrong shape in it is worse than none, for the same reason an unverified `path:line` is worse than
  no citation.
- **Evidence:** the op's docstring; `raw/G-SP-RING_20260904T145354Z.log` (output `(1, 4, 128, 128)`
  for `chunk_global=512`, `sp=4`, i.e. `chunk_local=128`).
- **Confidence:** high.
- **Blast radius:** documentation only.

---

### DEC-075 — The attention core is chosen in **one** function, `select_attention_core`, and every gate asserts it
- **Phase / module:** P8 / `tt/attention/prefill.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `DEC-072` makes the SP path a three-way choice (`dense` / `sp_bootstrap` / `sp_ring`),
  and Appendix B's final row is a mis-selection between two of the three.
- **Question:** inline the conditions in `attention_forward`, as the template does
  (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:182-191`), or factor them out?
- **Choice:** factor them into `select_attention_core(config, mesh_config, kv_cache, *, seq_len,
  cached_len) -> str`, called by `attention_forward` and **imported by the gates**.
- **Why:** a gate cannot assert which core ran if the condition lives inside the function under test.
  With the choice named, `G-MESH-KV` and `G-CHUNK-ATTN` both assert their expected core *before*
  spending three minutes measuring the wrong one, and the assertion is on the same expression the
  production path evaluates rather than a copy of it.
- **Evidence:** `raw/G-CHUNK-ATTN_20260904T150921Z.log.gz` — both arms log and assert their core
  (`sp_bootstrap` at chunk 1024, `sp_ring` at chunk 512).
- **Confidence:** high.
- **Falsifier:** if the two cores ever produced identical numbers the assertion would be the only
  thing distinguishing them — which is precisely why it exists. They do not: min K 0.9987994 vs
  0.9967119.
- **Blast radius:** `tt/attention/prefill.py`, every P8 mesh gate.

---

### DEC-076 — A `FABRIC_1D_RING` run must have the torus descriptor **exported before pytest starts**
- **Phase / module:** P8 / `tests/test_factory.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `TT_MESH_GRAPH_DESC_PATH` is read when metal initialises the control plane, which
  happens inside the `mesh_device` fixture — too late for a `conftest.py` to set it reliably, and
  setting it at import time would change every P5-P7 test's environment.
- **Question:** set it from Python, or require it and skip?
- **Choice:** require it. `requires_ring_fabric` skips with the exact `export` line in its reason.
- **Why:** the failure mode of getting this wrong is not a wrong number, it is a fabric init that
  aborts (or, per the recipe's claim, a hang). A skip with instructions is strictly better than a
  half-configured run.
- **Note:** on this galaxy the marker is inert, because `PREFILL_FABRIC` defaults to `1d`
  (`DEC-079`). It is kept for a torus-cabled machine.
- **Evidence:** `raw/G-FABRIC-MATRIX_20260904T142547Z.log` — the **aborted first sweep**, in which a
  five-level `dirname` walk (one short) made `TT_MESH_GRAPH_DESC_PATH` point at a nonexistent file
  and **every** case reported `std::filesystem::exists(mesh_graph_desc_path)` instead of what it was
  measuring, including two that "matched" their expectation for the wrong reason. That log is kept as
  evidence of the harness bug, and `_repo_root()` now asserts the descriptor directory exists.
- **Confidence:** high.
- **Blast radius:** the P8 gates' run instructions.

---

### DEC-077 — `SubmeshPool` quiesces on **both** sides of every hand-out
- **Phase / module:** P8 / `tests/test_factory.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `tt_metal/api/tt-metalium/mesh_device.hpp:296-305` requires a barrier between phases
  using overlapping submeshes and names `quiesce_devices()` (`:305`); nothing enforces it, and
  `G-TP-PARITY` compares `(1,1)` against five multi-device shapes **in one process**.
- **Question:** call `quiesce_devices()` at the seams by hand, or make forgetting impossible?
- **Choice:** a `SubmeshPool.use(shape)` context manager that calls `parent.quiesce_devices()`
  **before** creating/handing out the submesh and again **after** the phase ends, and caches
  submeshes by `(shape, offset)` so a five-shape sweep creates five.
- **Why:** the cost of forgetting is not a failed test, it is a machine-wide hang that poisons the box
  until `tt-smi -r` and turns every remaining gate into a false `FAIL`. A rule that has to be
  remembered at every call site will be forgotten at one. Quiescing on both sides makes "two live
  submeshes with no barrier between their phases" unreachable through the API.
- **Evidence:** the landmine measured directly —
  `raw/G-FABRIC-MATRIX_20260904T142819Z.log`: `overlap_1x2_then_1x8_no_quiesce` **hang** (246.3 s,
  box reset), `overlap_1x2_then_1x8_quiesce` **ok** (22.4 s, both phases bit-exact). And
  `raw/G-TP-PARITY_20260904T145701Z.log`: 6 submesh shapes, 24 phase transitions per module, 5 tests,
  67.6 s, no hang.
- **Confidence:** high.
- **Falsifier:** a hang inside `G-TP-PARITY` would falsify it. None occurred across ~100 hand-outs.
- **Blast radius:** every P8 test that opens a submesh.

---

### DEC-078 — Arm A of `G-KV-TP8` probes **V with RoPE on** and **K with RoPE off**
- **Phase / module:** P8 / `tests/unit/test_kv_cache_tp8.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the head→column claim must be **bit-exact** (recipe §2.5), and RoPE is
  position-dependent: rotating a labelled vector destroys the label.
- **Question:** how do you make a bit-exact head→column probe through the *full* model→cache path
  when that path includes RoPE?
- **Options considered:**
  1. Predict the post-RoPE values on the host. Requires reproducing the device's bf16 RoPE
     arithmetic exactly — a second implementation of the thing under test.
  2. Probe **V** with RoPE on: V is never rotated (`tt/attention/prefill.py` asserts it as an
     invariant at `G-ATTN`), so its lanes survive the whole path unchanged.
  3. Probe **K** with `transformation_mats=None`, which skips RoPE entirely.
- **Choice:** 2 **and** 3, as two parametrisations of one test.
- **Why:** the same `column_parallel` mapper places `k_proj` and `v_proj`, on the same axis with the
  same out-dim geometry, so the two probes together pin the mapping for both. K's *post-RoPE*
  correctness is arm B's (vs the fp32 golden) and `G-CHUNK-ATTN`'s.
- **The K probe's weight is head-uniform on purpose:** the loader `reverse_permute`s `k_proj`
  (`models/tt_transformers/tt/load_checkpoints.py:891`), which permutes rows *within* each head, so
  only a head-uniform label survives the swizzle. The V probe, which is not swizzled, carries the
  richer two-lane-block label the rotated-column control needs.
- **Evidence:** `raw/G-KV-TP8_20260904T144803Z.log` — `v_with_rope` and `k_without_rope` each
  8/8 columns bit-identical (`torch.equal`, `rtol=atol=0`).
- **Confidence:** high.
- **The gap, stated:** no bit-exact check exists for K's *post-RoPE* head→column placement. It is
  covered numerically by arm B (min K 0.9986432 over 32 layers) and by `G-CHUNK-ATTN`.
- **Blast radius:** `G-KV-TP8` arm A.

---

### DEC-079 — There is **no ring fabric** on this galaxy; P8 runs on `FABRIC_1D`
- **Phase / module:** P8 / `tests/test_factory.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** every `FABRIC_1D_RING` case in `G-FABRIC-MATRIX` failed.
- **Question:** is `FABRIC_1D_RING` available on one Blackhole Galaxy?
- **Finding:** **no**, and it is the cabling, not the configuration:
  - `FABRIC_1D_RING` + `single_bh_galaxy_torus_xy_graph_descriptor.textproto` (the only single-galaxy
    RING/RING descriptor) fails at `tt_metal/fabric/topology_mapper.cpp:544` with "Graph specified in
    MGD could not fit in the discovered physical topology ... Intra-mesh mapping failure for logical
    mesh 0 -> physical mesh 0: Mapping validation failed: **32 target node(s) are not mapped to any
    global node**";
  - it is **not** the channel policy: a copy of that descriptor with `policy: STRICT` changed to
    `RELAXED` fails identically;
  - `FABRIC_1D_RING` on a LINE/LINE or LINE/RING descriptor is refused a step earlier, at
    `tt_metal/fabric/mesh_graph.cpp:447-453` — "FabricConfig {} requests topology {} which requires
    more connectivity than MGD provides {}. FabricConfig can only restrict topology (e.g.,
    torus→mesh), not create new connections."
- **Choice:** `PREFILL_FABRIC` defaults to `1d` (`ttnn.FabricConfig.FABRIC_1D`), with `1d_ring` kept
  as an override for a torus-cabled machine.
- **Why:** it is the only fabric this machine can initialise, and it is sufficient — see `DEC-081`.
- **What the recipe says, and it is wrong here:** `BRINGUP_RECIPE.md:82-84`, "The Ring topology P8
  needs the torus descriptor; a Ring topology on a plain `FABRIC_1D` fabric **hangs** rather than
  erroring." On this box the torus descriptor cannot be used at all, and
  `ttnn.Topology.Ring` **collectives on `FABRIC_1D` do not hang** — they return bit-exact results at
  `(1,8)`/1 link, `(2,8)`/2 links and `(4,8)`/2 links on both axes.
- **Evidence:** `raw/G-FABRIC-MATRIX_20260904T142819Z.log` (5 ring-fabric cases `error`,
  `submesh_1x8_ring_on_fabric1d` `ok` and bit-exact),
  `raw/G-FABRIC-MATRIX-ADDENDUM_20260904T144233Z.log` (5/5 `ok`).
- **Confidence:** high on this machine; the recipe's claim may well hold on a torus-cabled galaxy,
  which is why the override stays.
- **Falsifier:** re-cabling, or `./build/test/tt_metal/tt_fabric/test_system_health` reporting the
  wrap links present — in which case the mapping failure would be a control-plane bug rather than a
  cabling fact.
- **Blast radius:** every P8 gate's `device_params`; `07_RISKS.md` R-030.

---

### DEC-080 — `G-KV-TP8` splits arm A in two: head→column at chunk 0, write offset without an attention core
- **Phase / module:** P8 / `tests/unit/test_kv_cache_tp8.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the first draft parametrised the head→column probe over `kv_actual ∈ {0, 128}`. The
  `128` case failed — correctly: `Attention.__call__` at `cached_len > 0` is **delta 3**, and on a
  `(1,8)` mesh there is no SP axis to run the ring core on, so the dense core refuses it.
- **Question:** weaken the refusal, move the probe to `(4,8)`, or split the arm?
- **Choice:** split. The full-path probe (projection → head split → RoPE → SDPA → write) runs at
  chunk 0 only; a second test drives `apply_qkv_projection` → `split_qkv_heads_prefill` →
  `write_kv_chunk` directly over three offsets `{0, 128, 256}` at TP=8.
- **Why:** the refusal is right and must not be weakened (recipe P7's instruction is explicit). The
  two claims are separable: head→column is about the *mapper*, the advancing offset is about the
  *write*, and only the second needs more than one offset. `G-CHUNK-ATTN` on `(4,8)` is where a
  cache-backed **core** runs.
- **Evidence:** `raw/G-KV-TP8_20260904T144803Z.log` — 24/24 (offset, column) blocks bit-identical,
  pad tail exactly 0. The earlier failure is in `raw/G-KV-TP8-ARMA_20260904T144614Z.log`.
- **Confidence:** high.
- **Blast radius:** `G-KV-TP8` arm A.

---

### DEC-081 — The collective topology is `Linear`, because the **ring SDPA** demands a wrap route the fabric lacks
- **Phase / module:** P8 / `tests/test_factory.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `G-SP-RING`'s first run, with `PREFILL_TOPOLOGY=ring` on `FABRIC_1D`.
- **Question:** `Ring` or `Linear` for `CCLManager.topology`, given `DEC-079`?
- **Finding:** plain collectives do not decide it — `ttnn.Topology.Ring` all-gathers are bit-exact on
  `FABRIC_1D` at every P8 shape. The **ring SDPA** does:
  `ttnn.transformer.ring_joint_scaled_dot_product_attention` under `Topology.Ring` asks the fabric for
  the SP axis's wrap-around route and aborts —
  `TT_FATAL @ tt_metal/fabric/fabric.cpp:174: forwarding_direction.has_value()`,
  "Could not find any forwarding direction from src (M0, D0) to dst (M0, D3)" — where D0→D3 is the
  4-device SP ring closing on itself. The identical call with `Topology.Linear` runs.
- **Choice:** `PREFILL_TOPOLOGY` defaults to `linear`.
- **Why:** it is the only topology under which the deployment's own attention core executes on this
  machine. `Ring` is not merely unnecessary here — it is unserviceable for the one op that needs a
  real ring, which is a sharper statement than `DEC-079`'s and is what settles the topology for the
  whole phase.
- **Measured cost of the choice: none that this phase can see.** `G-SP-RING` scores PCC 0.9996672 at
  **6.05x** its own floor under `Linear`, against the recipe's quoted 7.98x for the same op; and
  `G-TP-PARITY` is 0.9999733-1.0000000 across all five shapes. A `Ring`-vs-`Linear` A/B of the
  collectives alone is possible (both work) and is not done here: it would be a *performance*
  measurement, and perf is an explicit non-goal.
- **Evidence:** `raw/G-SP-RING_20260904T145034Z.log` (the `Ring` abort, with the full backtrace),
  `raw/G-SP-RING_20260904T145354Z.log` (`Linear`, 4/4 tests pass).
- **Confidence:** high.
- **Revisit if:** the machine is re-cabled for a torus, or the ring op gains a linear-route fallback.
- **Blast radius:** `CCLManager.topology` everywhere in P8; `07_RISKS.md` R-030, R-031.

---

### DEC-082 — `G-SP-RING`'s numeric control is a wrong `kv_cache_batch_idx`, because a wrong `kv_actual_isl` is refused
- **Phase / module:** P8 / `tests/unit/test_dense_sp_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the gate needs a negative control (§1.4). The obvious one — lie about
  `kv_actual_isl`, the chunk's position in the global sequence — turned out not to be numeric: the op
  **refuses** it.
- **Finding:** `kv_actual_isl=0` for a chunk that really starts at 512 aborts with
  `TT_FATAL @ ring_joint_sdpa_device_operation.cpp:278: new_actual_isl <= chunk_capacity`,
  "KV-pad-aware rotation expects current valid Q to fit in one fixed chunk. Got new_actual_isl=1024,
  chunk capacity=512" — because the op derives current valid tokens as `logical_n - kv_actual_isl`.
- **Choice:** keep that as a **structural** control (§1.4 counts a configuration that must refuse as
  one) and add a **numeric** control: a wrong `kv_cache_batch_idx`.
- **Why:** a gate needs at least one control the op *accepts* and gets wrong, or the assertion has
  never been exercised against a wrong answer. And `kv_cache_batch_idx` is the right choice on the
  merits: `dense_sp.py`'s fact 3 is that it must be `slot_idx * num_layers + layer_idx`, and passing
  the slot alone makes **every layer read layer 0's cache** — layer 0 correct by coincidence, layers
  1+ on stale K/V. That is the exact shape of bug a single-layer test cannot see, so the control
  populates two layers with different K/V and reads layer 1 both ways.
- **Evidence:** `raw/G-SP-RING_20260904T145354Z.log` — correct read PCC **0.9996770**, control read
  PCC **-0.00337**. The refused-`kv_actual_isl` control's `TT_FATAL` text is in the same log.
- **Good news worth stating:** an off-by-`actual_start` chunk cannot silently produce a wrong answer
  through this op. It aborts.
- **Confidence:** high.
- **Blast radius:** `G-SP-RING`.

---

### DEC-083 — `G-TP-PARITY` shards the sequence only for the **token-wise** modules
- **Phase / module:** P8 / `tests/unit/test_tp_parity.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:1791-1793`: "At SP > 1 the multi-device output is a token slice, so
  compare it against the corresponding slice of the `(1,1)` output".
- **Question:** does that hold for every module?
- **Finding:** only for the token-wise ones. `RMSNorm` and `MLP` act on each token row
  independently, so a sequence-sharded input does produce exactly the corresponding slice.
  `Attention` and `DecoderLayer` do **not**: the dense causal SDPA mixes tokens, so a
  sequence-sharded input makes each row block attend only itself and the output is not a slice of
  the single-device output at all.
- **Choice:** shard the sequence for `rms_norm` and `mlp` (and compare slices); **replicate** it for
  `attention` and `layer` (and compare the full output).
- **Why:** for the token-wise pair, the slice comparison is the direct proof of the CCL plan's central
  claim — collectives go on the TP axis only, so every module is SP-safe. For the other two,
  replication keeps the comparison a device-vs-device exactness claim while still running the TP
  collective on a 2-row and 4-row mesh at `num_links=2`, which is the transport the recipe's sentence
  is reaching for. The genuine SP attention core is `G-SP-RING`'s and `G-CHUNK-ATTN`'s.
- **Evidence:** `raw/G-TP-PARITY_20260904T145701Z.log` — `rms_norm` **1.0000000** at all five shapes
  (sequence-sharded, sliced), `mlp` 0.9999915 worst (sequence-sharded, sliced), `attention`
  0.9999917 and `layer` 0.9999733 worst (replicated). Control: the reference rolled by one TP shard
  scores **0.00307**.
- **Confidence:** high.
- **This is a deviation from the recipe's wording** and is reported as such.
- **Blast radius:** `G-TP-PARITY`.

---

### DEC-084 — `G-MESH-KV` drives the deployment path through `TtPrefillRuntime`, closing `R-029`
- **Phase / module:** P8 / `tests/galaxy_prefill_kv_pcc.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `07_RISKS.md` R-029 — `TtPrefillRuntime` had **never been instantiated**: the
  `tp == num_key_value_heads` equality forbids `(1,1)`, so `G-RUNTIME` could only audit it
  statically and no line of its happy path had executed.
- **Question:** drive `G-MESH-KV` through `tt/model.py` directly (simpler, fewer moving parts) or
  through the runtime?
- **Choice:** through `TtPrefillRuntime`, including its `compile()`.
- **Why:** P8 is the first moment the runtime *can* run, and a harness that reimplemented the chunk
  loop would have left the deployment object untested at exactly that moment — while adding a second
  chunk loop for the engine's contract to drift from. It also means `make_chunk_input`,
  `resolve_chunk_sizes`, `_build_indexed_rope`, `_resolve_kv` and the per-chunk argument checks all
  execute for real rather than against a stub.
- **Evidence:** `raw/G-MESH-KV-oneshot_20260904T150307Z.log` — `compile()` 18.9 s, one served chunk
  220.5 ms, 4643 tok/s, min K 0.9987994; `raw/G-MESH-KV-chunked512_20260904T150451Z.log.gz` — two
  chunks including a second at `actual_start=512`, `compile()` warming both, min K 0.9967119.
- **Confidence:** high.
- **What is still uncovered:** the six engine hooks still raise (`R-024`), so nothing on the
  migration, ack or trace paths has run. That is P10's.
- **Blast radius:** `G-MESH-KV`, `G-RACE`, `G-CHUNK-ATTN`, `G-SEMAPHORE`'s P8 arm; `R-029`.

---

### DEC-085 — `meta_head_index` is duplicated in the script and pinned by an equality test
- **Phase / module:** P8 / `tests/galaxy_prefill_kv_pcc.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `tests/galaxy_prefill_kv_pcc.py` is a **script** (recipe P8 names it as one, and
  `G-MESH-KV`'s mechanism is an exit code), so it must not import a pytest module — but it needs the
  HF→Meta head-dim permutation that `tests/unit/test_decoder_layer_vs_ref.py::_meta_head_index`
  already defines.
- **Question:** move the helper to `tests/test_factory.py`, or duplicate it?
- **Choice:** duplicate it in the script, and add
  `test_meta_head_index_does_not_drift_between_the_script_and_the_tests` asserting the two are equal
  at `head_dim ∈ {64, 128}`.
- **Why:** recipe §2.2 warns that two copies of a *floor helper* drift and then two gates disagree
  about what a floor is; the same argument applies here, and worse — a divergence would permute the
  golden one way in `G-MESH-KV` and another way in every unit gate, and **both** would look
  plausible. Moving it into `test_factory.py` would have been cleaner, but `test_factory.py` imports
  `pytest` at module scope, which is the thing the script must avoid. The equality test is the cheap
  half of the fix; the honest note is that the duplication is a wart.
- **Evidence:** `raw/G-CHUNK-ATTN_20260904T150921Z.log.gz` — the drift test passes.
- **Confidence:** medium. A better answer is a `pytest`-free helpers module; P9 may want it.
- **Revisit if:** P9's cleanliness gate objects to the duplication.
- **Blast radius:** `G-MESH-KV`, `G-CHUNK-ATTN`.

---

### DEC-086 — `G-CHUNK-ATTN` runs both arms from **one** runtime with two supported chunk sizes
- **Phase / module:** P8 / `tests/unit/test_chunked_attention_ring.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the gate compares the ring core against the SP bootstrap per layer, which means both
  must run in one process on the same weights.
- **Question:** two runtimes (two `Model`s, two `CCLManager`s, twice the weight memory) or one?
- **Choice:** one runtime, built with `chunk_size=512` and `additional_chunk_sizes=(1024,)`, so
  `rope_indexed` holds both tables and each `prefill_chunk` call passes the size it wants. At
  `max_seq_len=1024`, chunk 1024 selects `sp_bootstrap` and chunk 512 selects `sp_ring`.
- **Why:** identical weights on both sides is the whole point of a mutual-PCC gate — two builds would
  put the weight loader between the two arms. It also keeps one `CCLManager`, which is what
  `G-SEMAPHORE` requires, and halves device memory.
- **Evidence:** `raw/G-CHUNK-ATTN_20260904T150921Z.log.gz` — layer 0 mutual K **1.0000000** on both K
  and V, which is only possible if both arms saw byte-identical weights and byte-identical inputs.
- **Confidence:** high.
- **Blast radius:** `G-CHUNK-ATTN`.

---

### DEC-087 — `G-WEIGHTS`'s P8 arm hashes 32 shards of everything except the replicated vocab table
- **Phase / module:** P8 / `tests/unit/test_weight_loading.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the recipe asks for "every device tensor SHA-256-identical" at TP=8. The first
  attempt hashed all 32 shards of all 12 tensors and hit `pytest-timeout` at 300 s.
- **Question:** raise the timeout, or scope the sweep?
- **Finding:** `model.embed_tokens.weight` is **replicated, not TP-sharded** (`tt/embedding.py`,
  `DEC-024`), so each of the 32 devices holds the whole `[128256, 4096]` bf16 table — 1.05 GB each,
  **33.6 GB of device-to-host transfer per pass, twice**, to re-prove a tensor the mesh does not
  shard. It is 4x the cost of everything else combined.
- **Choice:** hash all 32 shards of all 11 other tensors (`lm_head.weight` included — it *is*
  column-parallel over the vocab), and for the embedding table hash the **first and last** device.
  Timeout raised to 2400 s for the arm.
- **Why:** for a replicated tensor the per-device claim is the `(1,1)` arm's claim repeated; the
  first-and-last pair is what keeps the *replication* itself falsifiable. Two further assertions make
  the scoping honest rather than convenient: the replicated tensors **must** have one distinct hash,
  and `q_proj` / `lm_head` **must** have more than one — so an arm that silently stopped sharding
  would fail rather than pass faster.
- **Evidence:** `raw/G-WEIGHTS-TP8_20260904T151945Z.log` — 354 device shards over 12 tensors, all
  identical; 8 tensors with **8 distinct** shard hashes (the 8 TP columns, replicated across the 4 SP
  rows — exactly the expected geometry), 4 replicated. The timed-out first attempt is
  `raw/G-WEIGHTS-TP8_20260904T151317Z.log`.
- **Confidence:** high.
- **The gap, stated:** the embedding table's shards on devices 1-30 are not hashed at TP=8.
- **Blast radius:** `G-WEIGHTS` (P8 ext).

---

### DEC-088 — P8 steps 5 and 6 required **no** code change, and here is why that is not an omission
- **Phase / module:** P8 / `tt/attention/operations.py`, `tt/mlp.py`, `tt/rms_norm.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** recipe P8 step 5 says "Turn on the collectives in the modules (they were written with
  the branch in place in P5)" and step 6 says "Enable the distributed RMSNorm branch **only if** the
  residual scheme is B". A phase that touches neither file needs to say so, or a reviewer cannot tell
  whether the steps were done or skipped.
- **Step 5 — the collectives were already on.** They are gated on `mesh_config.tp <= 1`
  (`tt/attention/operations.py::apply_allreduce`, and the same shape in `tt/mlp.py`), not on a flag,
  so they are a no-op at TP=1 and *live* at TP=8 with nothing to switch. `G-TP-PARITY` is the proof
  they run and are exact (worst 0.9999733 over five shapes, with the reference-rolled control at
  0.00307), and `G-SEMAPHORE`'s P8 arm confirms they draw from one `CCLManager`.
- **Step 6 — not applicable, because the residual scheme is A.** `DEC-025` took scheme A on the
  cost-equivalence argument, so the residual stream is full-emb replicated and
  `RMSNorm.is_distributed` stays `False`. Enabling it would be *wrong*, not merely unnecessary: the
  distributed norm expects an emb/TP-sharded input, and feeding it a replicated one is a mixed
  residual. `tt/rms_norm.py::_forward_distributed` is therefore still dormant in this package — and
  still carries the fix for the template's `stats`-passed-twice bug (`DEC-031`, `R-011`) so that
  whoever enables it does not inherit the `TypeError`.
- **Choice:** change nothing; record it.
- **Evidence:** `raw/G-TP-PARITY_20260904T145701Z.log`, `raw/G-SEMAPHORE_20260904T151056Z.log.gz`.
- **Confidence:** high.
- **Falsifier:** if `G-TP-PARITY` had shown TP=8 outputs equal to the TP=1 ones *without* a
  collective having run — i.e. if the reduce-scatter/all-gather pair were somehow a no-op — the
  shard-rotation control would still have discriminated but the mlp/attention numbers would have been
  exactly 1.0 rather than 0.99999x. They are not.
- **Blast radius:** none executable; the log's completeness.

---

### DEC-089 — P8 step 3's submesh parametrisation goes in a **new file**, not into the P5/P6 unit tests
- **Phase / module:** P8 / `tests/unit/test_tp_parity.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** recipe P8 step 3: "Add the submesh parametrisations to the P5/P6 unit tests: `(1,2)`,
  `(1,4)`, `(1,8)`, **`(2,8)`** and the target `(4,8)`."
- **Question:** parametrise `test_rms_norm_vs_ref.py`, `test_mlp_vs_ref.py`,
  `test_attention_vs_ref.py` and `test_decoder_layer_vs_ref.py` over five more shapes, or put the
  multi-device coverage in `G-TP-PARITY`'s own file?
- **Finding — the instruction as written is not executable on this machine, and it contradicts the
  recipe's own step 1.** The P5/P6 tests parametrise the repo-root `mesh_device` fixture
  (`conftest.py:554`), which calls `ttnn.open_mesh_device(MeshShape(*grid_dims))` at `conftest.py:661`
  — a **top-level** open. Step 1 of the same phase says that dies in fabric bring-up on this galaxy,
  and `G-FABRIC-MATRIX` measured it: `toplevel_1x8_fabric1d` and `toplevel_2x8_fabric1d` both
  `error`. So `@pytest.mark.parametrize("mesh_device", [(1,8)], indirect=True)` cannot be added to
  anything here.
- **Choice:** all five shapes plus the `(1,1)` reference live in `tests/unit/test_tp_parity.py`, as
  **submeshes** of one open `(4,8)` handed out by `SubmeshPool` (`DEC-077`), for all four modules
  `rms_norm` / `mlp` / `attention` / `layer`.
- **Why, beyond the mechanical blocker — three reasons the new file is also the better answer:**
  1. **A better instrument.** The recipe itself says to compare "**device outputs to each other**
     (not just each to torch) — sharper than PCC-vs-torch because it removes the reference's own
     error". A parametrised P5 test would compare each shape to *torch*, which is the weaker claim.
  2. **The P5/P6 thresholds do not transfer.** Every one was set against a `(1,1)` floor; at TP=8
     the row-parallel matmul's reduction order changes, so those numbers would be gating a different
     quantity under the same name.
  3. **Runtime.** Six shapes x four modules x the existing dtype/seq-len parametrisations would
     multiply four gates' cost for coverage one file gives in 68 s.
- **What is *not* covered by this choice, stated:** the P5/P6 gates themselves still run only at
  `(1,1)`, so their **floors and error ratios** remain single-card measurements. `G-TP-PARITY`
  establishes that the multi-device outputs equal them to 0.9999733+, which is what makes the
  single-card floors transferable — but it is an inference, not a re-measurement.
- **Evidence:** `raw/G-TP-PARITY_20260904T145701Z.log`; `raw/G-FABRIC-MATRIX_20260904T142819Z.log`
  for the blocker.
- **Confidence:** high.
- **This is a deviation from the recipe's step 3** and is reported as such, together with the
  step-1/step-3 contradiction.
- **Blast radius:** `G-TP-PARITY`; the P5/P6 test files are untouched.

---

### DEC-090 — The P8 regression was re-run after formatting, because the first run was mutated mid-flight
- **Phase / module:** P8 / hygiene
- **Date (UTC):** 2026-09-04
- **Trigger:** `black --line-length 120` was run over the new P8 files **while the first P8
  regression run was still executing**, reformatting seven of them.
- **What happened.** pytest had already imported and collected the modules, so the run in flight was
  executing the pre-format code. Its result is valid for that code and **not** for the tree as it
  now stands. This is recipe §0.2's rule — "never rename, move, or restructure while a session is
  live" — in a milder form than a rename, but the same class of mistake, and it was self-inflicted
  in the same way §0.2's three incidents were.
- **Choice:** keep the first run's log as the record of what it actually tested, and **re-run the
  regression on the final tree**, citing the second run in the ledger.
- **Why not just argue that `black` is semantics-preserving:** it is, and that argument is exactly
  the "verify a mutation with a smoke test" reasoning `LANDMINES.md` lists as a method trap. The
  regression costs one command; the argument costs a reviewer's trust.
- **Evidence:** both regression logs are in `raw/`, and the ledger's `P8-REGRESSION` row cites the
  post-format one.
- **Confidence:** high.
- **Kit note:** the formatting hooks (`black`, `isort`, `autoflake`) are documented in
  `LANDMINES.md` as things that "will block your commit" and that reformatting "moves the lines you
  just cited" — both true. What is not said, and cost this entry, is **when** to run them: before a
  long device run, not after. `isort` is additionally not installed in this `python_env`, so only
  `black` ran.
- **Blast radius:** the P8 regression row's provenance.

---

### DEC-091 — `G-CHUNK-ATTN` **skips** on a too-short golden trace, and the two per-layer JSONs collide
- **Phase / module:** P8 / `tests/unit/test_chunked_attention_ring.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `G-CHUNK-ATTN` needs a **1024**-token golden trace (two chunks of 512, so
  `chunk_local = 128` matches the ring `q_chunk_size`), while `G-CHUNK` (P7) was recorded against the
  **512**-token trace. Both read `$PREFILL_TRACE_DIR`, which the engine owns and which points at one
  directory.
- **Question:** assert the trace length, or skip?
- **Choice:** **skip**, with the regeneration command in the reason. Asserting made the per-phase
  regression *fail* whenever `PREFILL_TRACE_DIR` pointed at the 512-token trace — a harness fact
  presented as a defect, which is precisely the thing a gate must not do.
- **The second half of the problem, and it is not solved:** `tests/unit/test_attention_chunked_vs_ref.py`
  writes `raw/G-CHUNK_per_layer_pcc.json` and derives its shape from `metadata["n_tokens"]`, so
  running the regression at `s1024` **overwrites P7's evidence file with 1024-token content** while
  P7's ledger row cites a 512-token measurement. Measured, not theorised: the first P8 regression run
  did exactly that (`git diff` showed 211 changed lines in that JSON, plus a trailing-newline-only
  change in `G-MODEL_per_layer_pcc.json`).
- **Handling:** the two files are restored from the P7 commit after the final regression, and the
  incident is `07_RISKS.md` R-041. Not fixed by renaming P7's output, because that would break the
  `path`-style citation its ledger row carries — the fix belongs to whoever can update both, i.e. P9.
- **Why this matters beyond bookkeeping:** a per-phase regression that **writes into a previous
  phase's evidence** can silently invalidate a `PASS` recorded three phases ago, and nothing in the
  recipe's §1.2 raw-output rule anticipates it. Appendix C item 2 — "a gate with no raw log did not
  happen" — has an unstated corollary: a gate whose raw log was overwritten by a later phase did not
  happen either.
- **Evidence:** `raw/P8-REGRESSION_20260904T152156Z.log.gz` (the run that overwrote them, 196 passed),
  and `git diff` on the two JSONs.
- **Confidence:** high.
- **Blast radius:** `G-CHUNK`'s and `G-MODEL`'s evidence files; the regression's own reproducibility.

---

### DEC-092 — The two `TT_FATAL` controls use `try`/`except` rather than the `expect_error` fixture
- **Phase / module:** P8 / `tests/unit/test_dense_sp_vs_ref.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `G-SP-RING` has to record the **verbatim** `TT_FATAL` text for two refusals
  (`fp32_dest_acc_en=True` and a wrong `kv_actual_isl`), because the message *is* the measurement —
  the recipe asks for "the `TT_FATAL` text when `True` is refused" (`BRINGUP_RECIPE.md:1765-1767`).
  Everywhere else in this package a refusal is asserted with the repo-root `expect_error` fixture
  (`conftest.py:948`), which `LANDMINES.md` and `DEC-045` require.
- **Question:** `expect_error(RuntimeError, "<substring>")`, or catch and log?
- **Choice:** catch, log the full message through `loguru`, then assert on a metachar-free substring
  of it.
- **Why, three reasons:**
  1. **`expect_error` swallows the message.** It asserts and discards; the gate's job here is to put
     the text in `bringup_log/raw/` so a reader can see *which* assert fired. Both messages are now
     in the raw log verbatim.
  2. **`expect_error` matches its `message` as a regex**, not a substring (`R-014`, `DEC-045`), and
     these two messages are dense with metacharacters — `!kv_pad_rotation_enabled || use_streaming_compute`,
     `new_actual_isl <= chunk_capacity`, `(1,4,128,128)`. A literal would silently fail to match and
     the test would report `Regex pattern did not match` on correct code.
  3. **One of the two refusals was a surprise**, and that is exactly when you want the text rather
     than a boolean: the `kv_actual_isl` control was written expecting a numeric collapse and instead
     produced a refusal, which changed the gate's control design (`DEC-082`).
- **This is not a `try/except: pass`** (recipe §0 rule 5): the handler stores the message, the test
  logs it, and two assertions then require (a) that something was refused and (b) that it was refused
  by the specific assert the control names. A refusal from an unrelated cause fails the gate.
- **Evidence:** `raw/G-SP-RING_20260904T145354Z.log` — both messages appear in full, with their
  file:line.
- **Confidence:** high.
- **Blast radius:** `G-SP-RING`'s two structural controls.

---

### DEC-093 — `sp_ring_program_config`'s grid assertion took a parameter, because it could not fail
- **Phase / module:** P8 / `tt/attention/dense_sp.py`, `tt/attention/prefill.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** reviewing the phase's own new code against recipe §0 rule 5 ("no dead code") and §1.4
  ("a control that cannot fail is worse than no control, because it is recorded as evidence").
- **What was wrong.** `sp_ring_program_config` computed *both* sides of its safety assertion from the
  same expression:
  `ccl_offset_x = grid.x - 1` and `sdpa_grid_x = grid.x - 1`, then `assert sdpa_grid_x <= ccl_offset_x`.
  Tautological. It reads like the construction-time check `attention/config.py::validate_grid` is —
  the one that turns the P8-only ring landmine into a build-time failure — and it is not one.
- **Choice:** `sp_ring_program_config(mesh_device, *, ccl_core_grid_offset=None, ...)`, and
  `attention/prefill.py` passes `ccl_manager.ring_attention_ccl_core_grid_offset`. The assertion then
  compares **this file's** derivation of the SDPA grid against **`tt/ccl.py`'s** derivation of the CCL
  offset, which is a real check that the two have not drifted. Omitting the argument still falls back
  to the local derivation, and the docstring says that makes the check tautological.
- **Why not just delete the assert:** the constraint is real
  (`ring_joint_sdpa_device_operation.cpp:421`) and it is a P8-only failure mode — the same shape of
  landmine that motivated `validate_grid`. A check with a genuine second source is worth more than
  either a tautology or nothing.
- **Verified behaviour-neutral, not assumed:** `G-SP-RING` and `G-CHUNK-ATTN` were re-run after the
  change and reproduce every number exactly — PCC 0.9996672 / 6.05x, L1 mutual K 0.9999505, control
  worst 0.87279, 7/7 tests.
- **Evidence:** `raw/G-SP-RING-RECHECK_20260904T161926Z.log.gz`.
- **Scope note:** the final full regression (`raw/P8-REGRESSION_20260904T155300Z.log.gz`, 196 passed)
  predates this one-line change by 26 minutes. No test asserts on the signature, and the two gates
  that exercise the function were re-run; that is stated here rather than left for a reader to work
  out from timestamps.
- **Confidence:** high.
- **Blast radius:** `tt/attention/dense_sp.py`, `tt/attention/prefill.py`; `G-SP-RING`,
  `G-CHUNK-ATTN`.

---

### DEC-094 — `load_hf_config` reads the **bundled** config and refuses a disagreeing checkpoint
- **Phase / module:** P10 / `tt/runners/adapters/llama.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** implementing the first abstract method. The template does
  `AutoConfig.from_pretrained(PREFILL_HF_MODEL)`
  (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:68`), and this package refuses config
  *objects* outright (`tt/model_config.py:101-107`, `DEC-019`).
- **Question:** where do the engine's dimensions come from, given that `ModelArgs` is supposed to be
  the one place a config value is read (`DEC-019`) and the engine wants an attribute-shaped, mutable
  object?
- **Options considered:**
  1. `AutoConfig.from_pretrained`, like the template. Reintroduces recipe P1 trap 1 (`R-005`) for
     anything downstream that `getattr`s a moved attribute, and pulls `transformers` into the
     adapter's import chain, which `G-ADAPTER` gates against.
  2. Read `<PREFILL_HF_MODEL>/config.json` with `json` and wrap it. Import-light and correct, but
     creates a **second** reader of config.json alongside `ModelArgs.load_bundled_config`.
  3. Always use `ModelArgs.load_bundled_config()`, ignoring `PREFILL_HF_MODEL` entirely. One reader,
     but silently ignores an env var the engine's contract says overrides `hf_model_default`, and
     prints a path it did not read (`prefill_runner.py:375`).
  4. Option 2 **plus an equality refusal**: read the pointed-at file, compare it to the bundled copy,
     and raise naming the differing keys if they disagree.
- **Choice:** option 4. The returned object is `LlamaHfConfig`, an attribute view whose `.dims` is
  the raw dict every module downstream takes.
- **Why:** it honours the env var, keeps one *authoritative* source (a second reader that can only
  agree is not a second answer), and turns the dangerous case — a checkpoint whose dims differ from
  the one every threshold and the whole weight cache were built against — into a startup
  `ValueError` naming the keys instead of a silently different model.
- **Evidence:** `DEC-001` (the bundled copy is byte-identical to the staged checkpoint's, asserted
  by `tests/unit/test_reference_model.py`); `models/demos/common/prefill/adapter.py:116`
  (`hf_model_default`: "config.json dir; PREFILL_HF_MODEL overrides");
  `G-ADAPTER::test_load_hf_config_refuses_a_disagreeing_checkpoint_config` measures both halves.
- **Confidence:** high.
- **Falsifier:** a deployment that legitimately serves a differently-shaped Llama through this
  adapter — at which point the dims must come from the checkpoint and every threshold is re-measured
  anyway.
- **Revisit if:** a second checkpoint is added to this package.
- **Blast radius:** `tt/runners/adapters/llama.py`, `G-ADAPTER`, `G-REQUEST`, `G-MOCK-MIG`.

---

### DEC-095 — `weight_cache_path` mirrors **this package's** layout, not the engine's convention
- **Phase / module:** P10 / `tt/runners/adapters/llama.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the engine's own reference adapter returns
  `$PREFILL_TTNN_CACHE/{name}_{arch}_{N}dev/{sp}x{tp}`
  (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:89`), while every cache this box holds
  was written to `$TT_CACHE_PATH/tensor_cache_bfp8_<sp>x<tp>` by `tt/model_config.py:250`.
- **Question:** which layout does the adapter return?
- **Options considered:**
  1. The engine's convention. Consistent with the other adapters; means the first runner start
     re-tilizes and re-writes 15 GB of weights into a new tree, and P8's populated cache — the one
     `G-WEIGHTS`'s TP=8 arm proved SHA-256-identical on a cache-only rebuild — is dead.
  2. This package's layout, i.e. mirror `ModelArgs.weight_cache_path`. The doc's instruction taken
     literally: "Mirror the layout the cache-populate run wrote so the runner reads the same files"
     (`ADDING_A_PREFILL_MODEL.md:68`).
  3. Change `ModelArgs.weight_cache_path` to the engine's convention and repopulate. Touches a P6
     file and invalidates a P8 gate's evidence for a cosmetic gain.
- **Choice:** option 2, with `PREFILL_TTNN_CACHE` accepted as the root ahead of `TT_CACHE_PATH`, and
  `G-ADAPTER` asserting the adapter's answer **equals** `ModelArgs`' for the same root and mesh.
- **Why:** the doc's own words, and the cache-populate run is ours. The equality assertion is what
  keeps the duplicated path string from drifting; without it this would be a copy-paste.
- **What it costs:** the path does not carry the model name, so two models sharing a `TT_CACHE_PATH`
  would collide. Stated rather than fixed: the mesh shape and dtype are in the path, the root is
  per-model on this box (`~/.cache/llama31_8b_d_p`), and changing it now would invalidate P8's cache.
- **Evidence, and the decisive half is measured on device.**
  `raw/G-MESH-KV-oneshot_20260904T150307Z.log` line 33 shows P8 *writing*
  `/home/mstojkovic/.cache/llama31_8b_d_p/tensor_cache_bfp8_4x8`;
  `G-ADAPTER::test_weight_cache_path_mirrors_the_packages_own_layout` asserts the adapter returns
  the same path; and `raw/G-REQUEST-runner_20260904T173723Z.log.gz` shows the runner *reading* it —
  **290 `Loading cache` lines and 0 `Generating cache` lines**. Under the engine's own convention
  every one of those 290 would have been a regeneration on the first start, which is the whole
  argument, and it is a count rather than an inference.
- **Confidence:** high.
- **Falsifier:** a deployment that shares one `PREFILL_TTNN_CACHE` across models, where the missing
  `{name}` segment would make two models read each other's tensors — the failure would be
  "one layer runs on garbage" (Appendix B), which is why the root is per-model here.
- **Revisit if:** this package is deployed alongside another under one cache root.
- **Blast radius:** `tt/runners/adapters/llama.py`, `G-ADAPTER`, every runner start.

---

### DEC-096 — `PREFILL_KV_ONLY_LAST_LAYER` is **ignored with a warning**, not refused
- **Phase / module:** P10 / `tt/runners/adapters/llama.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `PrefillRunParams.kv_only_last_layer` is a field this runtime has no use for, and the
  engine defaults it **on** (`prefill_runner.py:80`: `PREFILL_KV_ONLY_LAST_LAYER` defaults to `"1"`).
- **Question:** recipe §0 rule 5 says nothing may be silently unimplemented, and this runtime writes
  every layer's KV regardless. Refuse, ignore, or ignore loudly?
- **Options considered:**
  1. **Refuse** when it is set. Correct by the letter of the rule, and it makes the engine's
     **default** configuration unrunnable — every operator would have to set
     `PREFILL_KV_ONLY_LAST_LAYER=0` to serve this model at all.
  2. **Ignore silently**, as the field is not read. Exactly the "knob a caller can believe in
     wrongly" that `DEC-062` argued against for `skip_lm_head`.
  3. **Ignore, and warn at build time**, naming what this runtime does instead.
- **Choice:** option 3.
- **Why:** writing every layer's KV is a *superset* of what the flag asks for, so ignoring it cannot
  produce a wrong answer — and it is what both consumers of the cache actually require: the
  producer's read-back PCCs every layer (`prefill_producer.py:565`) and migration copies every
  layer's rows. So the flag can only ever ask for less coverage, never for different data. Option 1
  would trade a real cost (an unrunnable default) for no correctness gain.
- **Evidence:** `prefill_runner.py:80` (the default), `:491` (`kv_only_last_layer=is_last_rank and
  KV_ONLY_LAST_LAYER`), `prefill_producer.py:565` (the reader's per-layer loop). The warning text is
  in `tt/runners/adapters/llama.py`'s `build_runtime`.
- **Confidence:** high.
- **Falsifier:** a memory-constrained configuration where the cache cannot hold every layer — at
  which point the flag has to change the *allocation*, not just the writes, and `allocate_kv_cache`
  is where it would land.
- **Revisit if:** `num_users` or `max_seq_len` grows enough that a 32-layer cache does not fit.
- **Blast radius:** `tt/runners/adapters/llama.py`; nothing numerical.

---

### DEC-097 — `Topology.Linear` is pinned in the adapter, not read from an env var
- **Phase / module:** P10 / `tt/runners/adapters/llama.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `TtPrefillRuntimeConfig.topology` has to be set, `PrefillRunParams` carries no
  topology field, and the reference adapter fills the gap by reading `PREFILL_TOPOLOGY` from
  `os.environ` (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:140`) — which recipe P10
  step 1 forbids ("Read knobs from `params`, **never** from `os.environ`").
- **Question:** how does the deployment topology reach the runtime?
- **Options considered:**
  1. Read `PREFILL_TOPOLOGY`, like the template. Violates the recipe's instruction, and adds a fifth
     undocumented-until-P9 env var to a package that has four (`DEC-023`).
  2. Default `TtPrefillRuntimeConfig.topology` and pass nothing. The dataclass default is
     `Topology.Ring`, which on this galaxy **aborts** the ring SDPA — so the runner would die inside
     attention on its first chunk.
  3. Pin `Topology.Linear` explicitly in the adapter, with the measurement as the comment.
- **Choice:** option 3.
- **Why:** on this machine the topology is not a choice. `FABRIC_1D_RING` cannot be initialised at
  all, and `ttnn.transformer.ring_joint_scaled_dot_product_attention` under `Topology.Ring` asks the
  fabric for the SP axis's wrap route and aborts at `tt_metal/fabric/fabric.cpp:171`
  ("Could not find any forwarding direction from src (M0, D0) to dst (M0, D3)") — measured by
  `G-FABRIC-MATRIX` and recorded as `DEC-079`/`DEC-081`, `R-030`/`R-031`. A knob whose only legal
  value is one value is not a knob; making it an env var would invite someone to set the value that
  crashes.
- **What it costs:** a torus-cabled machine would need a code change rather than an export. Stated
  in `07_RISKS.md` R-031, whose owner is a future phase on different hardware.
- **Evidence:** `06_GATES.md` `G-FABRIC-MATRIX` and `G-SP-RING`; `tests/test_factory.py:194-235`
  (the same coupling, behind one variable, for the package's own tests).
- **Confidence:** high — measured, twice, on this box.
- **Falsifier:** a machine whose torus descriptor maps, where `Ring` would be both available and
  faster.
- **Revisit if:** the deployment moves to a torus-cabled galaxy, or `PrefillRunParams` grows a
  topology field.
- **Blast radius:** `tt/runners/adapters/llama.py`; `G-REQUEST`, `G-MOCK-MIG`.

---

### DEC-098 — No cache-only build path in the adapter: it always loads the real checkpoint
- **Phase / module:** P10 / `tt/runners/adapters/llama.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the template gates its state-dict load behind an env var
  (`GPT_OSS_WEIGHTS_FROM_CACHE=1`, `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:147`),
  and this package's `Model` supports a cache-only build with an empty state dict (`G-WEIGHTS`).
- **Question:** should the adapter skip the ~15 GB safetensors read when the weight cache is
  populated?
- **Options considered:**
  1. A new env var, mirroring the template. Costs a `DEC` and a `README` entry (recipe §1.3), and
     adds a fifth package env var whose wrong setting is a silent failure mode.
  2. **Derive** it: skip the load when `weight_cache_path` looks populated. A *partially* populated
     cache then reaches the loader with no source for the missing tensor. The loader does fail loud
     (`tt/model_config.py`, `DEC-048`), but the trigger becomes "how full is a directory", which is
     not a property anything asserts.
  3. Always load the checkpoint.
- **Choice:** option 3.
- **Why:** the cost is bounded and paid once per runner start (measured below), and neither
  alternative buys correctness. Option 2's failure mode in particular is the one Appendix B calls
  "cache-only build silently wrong".
- **What it costs — and the measurement contradicted the assumption behind the question.** It is
  **46 ms**, not the tens of seconds this decision was weighing. `ModelArgs.load_state_dict` reads
  the shards through `safetensors`, which **memory-maps** them, so `load_state_dict` returns almost
  immediately and a tensor's bytes are faulted in only when something touches them — and with the
  weight cache populated (`DEC-095`) nothing does: the same run logged **290 `Loading cache` lines
  and 0 `Generating cache` lines**, so no cached tensor ever consulted the state dict. The whole of
  `build_runtime` -> `setup complete` is **13.2 s**, and most of that is `compile()`'s two warm-up
  chunks (17:37:47.457 -> 17:37:49.446 for the second alone) plus the rope tables.
  So option 1's env var and option 2's derivation would both have been complexity bought to avoid a
  cost that does not exist on the populated-cache path. It *would* exist on an empty cache — where
  the read is unavoidable anyway, because that is the run that populates it.
- **Evidence:** `raw/G-REQUEST-runner_20260904T173723Z.log.gz` — first line 17:37:25.400,
  `build_runtime` logging the weight load at 17:37:36.796, the runtime constructed at 17:37:36.842
  (**46 ms** later), `setup complete` at 17:37:50.007; 290 `Loading cache` / 0 `Generating cache`.
  `G-ADAPTER::test_build_runtime_refuses_a_missing_checkpoint` asserts the refusal when `HF_MODEL` is
  unset, so the failure is a named `ValueError` rather than a `NoneType` path join.
- **Confidence:** high.
- **Falsifier:** a deployment that restarts runners often enough for 55 s to matter.
- **Revisit if:** perf work starts, or the checkpoint moves off local disk.
- **Blast radius:** `tt/runners/adapters/llama.py`; runner start-up time only.

---

### DEC-099 — The gpt-oss KV-chunk-table builder is **imported**, not copied
- **Phase / module:** P10 / `tt/runners/kv_chunk_table.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** writing the block-cyclic address walk. Recipe P10 step 4 says to write
  `tt/runners/kv_chunk_table.py`; agent-contract rule 4 says "Reuse means *import*, not copy-paste. A
  copy-paste is a `DEC` with a justification."
- **Question:** copy `models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:66`'s 140-line bank walk,
  or import it?
- **Options considered:**
  1. **Copy** and adapt. What the recipe's file list implies and what every other package in the tree
     has done. Two copies of the same DRAM address arithmetic, each gated by its own device test,
     free to drift in between.
  2. **Import** and wrap. Legal only because P5.6 deliberately kept this package's cache structurally
     identical to gpt-oss's *for this purpose*: the same `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`,
     the same `[1, 1, 32, head_dim]` `NdShardSpec` with `ROUND_ROBIN_1D`, the same user-major
     `slot = user * num_layers + layer` packing (`tt/attention/kv_cache.py:17-30`). `head_dim` and
     `num_kv_heads` are already parameters there.
  3. Move the builder into `models/demos/common/prefill/`. The right long-term home, and out of scope:
     this session may not add files outside the package.
- **Choice:** option 2, with two guards. `_assert_layout_still_shared()` raises if the two packages'
  block constants ever diverge, and `G-KV-TABLE` reads every resulting address back over UMD and
  compares **bit-exactly** to the live cache — so an upstream change to that walk fails here as a
  wrong address, not as a slightly worse PCC.
- **Why:** P5.6 paid for this on purpose (`bringup_log/03_OUTLINE.md` §2.7: "that is what lets P10
  reuse the producer's existing packed-GQA read-back instead of writing a fourth reader" — the same
  argument applies to the writer). Declining to reuse would make P5.6's constraint pointless.
- **What it costs:** a cross-package dependency on another model's *runners* module, which is a
  layering smell, and an upstream edit lands here unannounced. Recorded as `07_RISKS.md` R-045 with
  the fix named (promote it to `models/demos/common/prefill/`, which needs an owner outside this
  package — `H7`).
- **Evidence:** `G-KV-TABLE`: 2048 chunks bit-identical over UMD at two block-cyclic periods, five
  discriminating controls, protobuf round trip with 16 configs.
- **Confidence:** high.
- **Falsifier:** either package changing its DRAM shard geometry — which the assertion catches at
  build time on our side and `G-KV-TABLE` catches on theirs.
- **Revisit if:** the builder is promoted to `common/`, or this cache's layout changes.
- **Blast radius:** `tt/runners/kv_chunk_table.py`, `tt/tt_prefill_runtime.py::build_kv_chunk_table`;
  `G-KV-TABLE`, `G-MOCK-MIG`.

---

### DEC-100 — Serialized through `serialize_prebuilt_kv_chunk_table`, not the recipe's named helper
- **Phase / module:** P10 / `tt/runners/kv_chunk_table.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** recipe P10 step 4 says to implement the migration hooks "using `serialize_kv_chunk_table`
  from `models/demos/common/prefill/runners/migration.py`".
- **What is wrong with that instruction:** `serialize_kv_chunk_table` (`migration.py:220`) *builds* a
  **single-config** table — it constructs one `KvChunkAddressTableConfig`, hands it to a
  `table_builder(config=..., chunk_size_bytes=..., num_users=...)` callback, and serializes the
  result. This model's table has `2 x num_kv_heads = 16` configs (K head 0..7, then V head 0..7),
  which that signature cannot express. The reference GQA implementation does not use it either; it
  calls `ttnn.experimental.disaggregation.export_to_protobuf_file` directly
  (`models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:206`).
- **Options considered:**
  1. Follow the recipe literally. Impossible without collapsing 16 configs into one, which would
     lose the head->config mapping the migration contract is built on.
  2. Call `export_to_protobuf_file` directly, like the template. Works, and writes the file
     **in place** — a reader polling for the path (and `prefill_producer.py:128-134` polls exactly that
     way) can import a half-written table.
  3. `serialize_prebuilt_kv_chunk_table` (`migration.py:240`) — the *same module's* entry point for a
     table that is already built. It routes through `_serialize_table_to_path` (`migration.py:39-42`),
     which writes `<path>.tmp` and `os.replace`s it, so the publish is atomic.
- **Choice:** option 3.
- **Why:** it satisfies the recipe's actual intent (use the shared helper rather than hand-rolling
  the boilerplate) with the one function in that module that fits a multi-config table, and it fixes a
  real race the template has. `G-KV-TABLE` asserts the `.tmp` file does not survive.
- **Evidence:** `migration.py:220-237` (the single-config signature), `:240-246` (the prebuilt one),
  `:39-42` (the atomic replace); `prefill_producer.py:127-134` (the polling reader);
  `G-KV-TABLE::test_protobuf_round_trip_preserves_every_lookup_and_every_config_name`.
- **Confidence:** high.
- **Falsifier:** none for this model; if `serialize_kv_chunk_table` grows a multi-config form, use it.
- **Revisit if:** `migration.py` gains a multi-config builder.
- **Blast radius:** `tt/runners/kv_chunk_table.py`; `G-KV-TABLE`, `G-MOCK-MIG`.

---

### DEC-101 — `G-ADAPTER`'s import budget is a **ceiling on the import chain**, not a perf target
- **Phase / module:** P10 / `tests/unit/test_prefill_adapter.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:1938` requires "adapter import is **measured** — time it and assert
  no heavy module landed in `sys.modules`". A time assertion needs a number, and a number not read
  from a file needs a `DEC` (§1.3).
- **Question:** what should the threshold be, and against what?
- **Choice:** `1.0 s` in a **cold subprocess**, with the measured value logged; the load-bearing
  assertion is the `sys.modules` one.
- **Why this number:** it is not a tuned budget, it is a separator. Importing `ttnn` alone costs
  seconds *and* opens the 32-device cluster on this box (visible in every gate log), and `torch` is
  ~1 s by itself, so a run that walked the heavy chain cannot come in under a second. Measured:
  **40 ms** for the adapter alone. Anything within an order of magnitude of that is proof; the
  threshold is set 25x above the measurement so a slower machine or a cold page cache does not turn
  the gate red for no reason.
- **Why a subprocess:** in-process the assertion is vacuous — pytest has already imported `torch`
  and `ttnn` before the first test runs, so `"torch" in sys.modules` is always true. The probe gets
  a **negative control**: the same subprocess with `tt/model_config.py` added must report `torch`
  and `ttnn`, which it does. Without that control, "no heavy module found" and "the probe looks in
  the wrong place" are the same observation (`R-016`'s shape).
- **Evidence:** `raw/G-ADAPTER_20260904T173636Z.log`
  (`test_adapter_import_is_cheap_and_pulls_no_device_stack`, and the control immediately after).
- **Confidence:** high.
- **Falsifier:** a machine where `import torch` is under a second — then the time half is
  meaningless and the `sys.modules` half still holds, which is why the latter is the assertion that
  matters.
- **Revisit if:** the adapter grows a legitimate module-scope dependency.
- **Blast radius:** `G-ADAPTER`.

---

### DEC-102 — "the code does not mention X" assertions strip docstrings before searching
- **Phase / module:** P10 / `tests/unit/test_prefill_adapter.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** three `G-ADAPTER` assertions failed on their **own prose**. This package's docstrings
  cite the traps they avoid by name — `AutoConfig`, `get_num_devices` — so a substring search over
  `inspect.getsource` matches the *warning* as readily as the offence. It is the same defect shape as
  the repo's own `prefer-expect-error` hook, which fires on a docstring explaining that the file uses
  `expect_error` instead (`LANDMINES.md`, "Repo hooks").
- **Question:** weaken the prose, or make the check see only executable code?
- **Options considered:**
  1. **Reword the docstrings** to avoid the forbidden strings. What the repo hook's own guidance
     recommends, and it makes the documentation worse to satisfy a test: the whole value of naming
     `AutoConfig` in `load_hf_config`'s docstring is that the next reader learns why it is absent.
  2. **Search only the executable code**: `ast.parse`, drop every docstring node, `ast.unparse`.
     Comments go too, since `unparse` does not emit them.
- **Choice:** option 2, as `_executable_code(obj)`.
- **Why:** the claim being tested is about what the module *does*, and that is exactly what
  `ast.unparse` of a docstring-stripped tree contains. It also made the `rope_theta` assertion
  sharper: the naive version wanted the string absent, but `rope_theta` legitimately appears as a
  **dict key** in `CONFIG_JSON_KEYS`, so the assertion was rewritten to find every three-argument
  `getattr` — which *is* the trap — and require its name set to be `{"max_seq_len"}` (the one
  attribute the engine genuinely assigns later).
- **Evidence:** the three failures, in this session's first `G-ADAPTER` run; the passing form in
  `raw/G-ADAPTER_20260904T173636Z.log`.
- **Confidence:** high.
- **Falsifier:** a heavy import hidden inside an `exec` or a string, which neither form catches; the
  subprocess `sys.modules` probe does.
- **Revisit if:** P9 adds more "the code must not contain X" checks — they should use this helper.
- **Blast radius:** `G-ADAPTER` only.

---

### DEC-103 — `G-LOOPBACK` is **out of scope**, not blocked
- **Phase / module:** P10 / `06_GATES.md`, `07_RISKS.md`
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:1968-1975` gives three legal outcomes for this gate: `PASS`,
  out-of-scope with a `DEC` **and a named residual risk**, or `BLOCKED` with a reason. It also says
  "Do not fake it."
- **Question:** which of the three, and on what grounds?
- **What the gate needs, itemised:** the tt-llm-engine binaries `migration_endpoint` and
  `migration_worker` built against this tt-metal tree, a `_migration_client*.so` on
  `PREFILL_MIGRATION_CLIENT_DIR`, three shared-memory queue trios, and an MPI launcher able to place
  two worker slots on this host (`PREFILL_MIGRATION_TESTING.md:456-493`, `:587-596`). None of it is
  in this repository.
- **Options considered:**
  1. `BLOCKED`. Honest about the missing binaries, but wrong about *whose* gate it is: a red
     `G-LOOPBACK` is overwhelmingly likely to be an engine or launcher failure, not a failure of this
     model. `HUMAN GATE H4` exists to ask exactly that question — "ask whose bug a red gate would
     be" (`BRINGUP_RECIPE.md:1857`, §0.3) — and the doc itself says the gate "verifies the *engine's*
     model-agnostic byte copy, not this model".
  2. **Out of scope by decision**, with the residual gap enumerated as a named risk.
  3. Simulate it — copy bytes host-side and call it a loopback. Explicitly forbidden.
- **Choice:** option 2. Recorded as `R-043`, which enumerates every property that stays unproven.
- **Why:** it is the outcome the recipe's own reasoning points to, and it is what
  `WHY_THESE_EXAMPLES.md` cites the integration coverage table for ("which is how `G-LOOPBACK` got
  correctly scoped out instead of being recorded as a blocker"). Crucially, the *input* to that byte
  copy — the address table it reads — is proved bit-exactly by `G-KV-TABLE`, and the doc's own hook
  table says `dst-bytes` needs **no** model-specific surface
  (`PREFILL_MIGRATION_TESTING.md:544`). So what is unproven is the transport, and the transport is
  not ours.
- **What stays unproven (R-043):** the `MigrationLayerClient` attach, the `WORKER_READY` handshake
  (`migration.py:270-280`), `publish_serialized_table_and_wait_ready` — which this package's code
  never calls, since the engine owns it — and whether a real worker can read the addresses this table
  publishes. Also unproven: that `kv_migration_base_address` returns something the engine's stage
  gather is happy with, since only the real path calls it.
- **Evidence:** `PREFILL_MIGRATION_TESTING.md:14` ("+ tt-llm-engine binaries"), `:456-461`, `:534-550`.
- **Confidence:** high on the scoping; the residual gap is real and stated.
- **Falsifier:** a loopback run that fails on something in *this* package — most plausibly the
  address table, which is why `G-KV-TABLE` gates it bit-exactly rather than by PCC.
- **Revisit if:** the tt-llm-engine binaries become available on this box.
- **Blast radius:** `G-LOOPBACK`, `R-043`; nothing in `tt/`.

---

### DEC-104 — The producer's packed-GQA read-back branch is generalised (shared code, outside the package)
- **Phase / module:** P10 / `models/demos/common/prefill/runners/prefill_producer.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** recipe P10 step 5. The device-less KV reader that powers `PREFILL_PRODUCER_CHECK_PCC`
  is **not** adapter-dispatched: it branches on `ADAPTER.name` inside
  `_read_slot_kv_and_check_pcc` (`prefill_producer.py:511`) and falls through to
  `_read_slot_kv_and_check_pcc_mla` (`:694`), which decodes a merged MLA latent+rope row. Without a
  branch, `G-MOCK-MIG` would PCC **plausible but wrong bytes**.
- **Question:** how little shared code can this be, and in what shape?
- **Options considered:**
  1. **A fourth reader** for this model. The doc's older wording implies it
     (`ADDING_A_PREFILL_MODEL.md:241`, "a third layout needs a branch"), and it is wrong here: the
     existing `_read_slot_kv_and_check_pcc_gpt_oss` is *already* this cache's layout — plain packed
     K/V, block-cyclic on SP, one KV head per TP column, HF-layout golden — because P5.6 kept
     gpt-oss's geometry deliberately (`tt/attention/kv_cache.py:17-30`).
  2. **Add `"llama31_8b_d_p"` to the existing `==` check** as a second `if`. Two nearly identical
     lines, and the next model makes three.
  3. **Generalise the check** to a named tuple of models and rename the function to what it reads.
- **Choice:** option 3, exactly as recipe P10 step 5 directs ("generalise the name check rather than
  duplicating the function, and make the reader's log line name `ADAPTER.name` instead of a
  hard-coded model"). Three edits, all in one function's neighbourhood:
  `_PACKED_GQA_MODELS = ("gpt_oss_d_p", "llama31_8b_d_p")` with a comment stating what a new entry
  must satisfy; `_read_slot_kv_and_check_pcc_gpt_oss` -> `_read_slot_kv_and_check_pcc_packed_gqa`;
  and the summary log line now reads `{ADAPTER.name} packed-GQA KV PCC`.
- **Why the rename is safe:** `grep` finds no other reference to the old name in the repository —
  `migration_driver.py` reaches this layer only through `_read_slot_kv_and_check_pcc` (the
  dispatcher) and `_read_kv_slice`, both unchanged.
- **What was deliberately NOT changed:** `_read_kv_slice` still imports
  `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK` from `models.demos.minimax_m3` (`prefill_producer.py:531`),
  i.e. one model's constant reads every model's cache. All three packages set it to 32, so it is
  latent, and fixing it is a shared-code change with no gate of its own in this phase. Recorded as
  `R-047`; our side asserts our value against the template's
  (`tt/runners/kv_chunk_table.py::_assert_layout_still_shared`).
- **Why the rotary permutation needs no work:** the reader permutes the **golden** HF -> Meta with
  `perm[m] = half * (m % 2) + (m // 2)` over `ROTARY_DIM`, defaulting `ROTARY_DIM` to `HEAD_DIM`
  (`prefill_producer.py:552-557`). For Llama the whole head is rotated, so that default is right and
  the permutation is byte-identical to this package's own `meta_head_index`
  (`tests/galaxy_prefill_kv_pcc.py:204`) — the same function `G-MESH-KV` scores against.
- **Evidence:** `G-MOCK-MIG`'s per-layer numbers agreeing with `G-KV-TABLE` and `G-MESH-KV`; the
  reader's log line naming `llama31_8b_d_p` in `raw/G-MOCK-MIG-producer_*.log`.
- **Confidence:** high.
- **Falsifier:** a packed-GQA model whose golden is stored in Meta order, or whose rotary dim is
  narrower than its head dim — either would need the branch split after all.
- **Revisit if:** a third packed-GQA model is registered, or the constant in `R-047` diverges.
- **Blast radius:** `models/demos/common/prefill/runners/prefill_producer.py` (shared),
  `models/demos/gpt_oss_d_p`'s own Gate-1 runs, `G-MOCK-MIG`.

---

### DEC-105 — `G-KV-TABLE`'s probe fixture is **function**-scoped, because the mesh is
- **Phase / module:** P10 / `tests/unit/test_kv_chunk_table.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the first draft cached the (expensive) probe write in a module-scoped fixture. Both
  bit-exactness tests then failed with
  `TT_FATAL @ tt_metal/distributed/mesh_device.cpp:845: id < mesh_command_queues_.size()` /
  "cq_id 0 is out of range", from inside `ttnn.to_torch`.
- **What was actually wrong:** the repo's `mesh_device` fixture is **function**-scoped, so the mesh is
  closed and reopened between tests. A cached device tensor then belongs to a closed mesh, and the
  next `from_device` on it finds no command queues. The message names neither the fixture nor the
  stale tensor.
- **Why it is worth a `DEC` rather than a silent fix:** the failure was *partial*. The tests that
  only read **addresses** — `test_head_to_config_to_chip`, `test_table_geometry` — passed, because a
  stale tensor's `buffer_address()` still returns a plausible number and the table builder is pure
  host arithmetic. So a cross-fixture cache produces a suite where the addressing tests are green and
  meaningless. Had the bit-exact tests not existed, the gate would have "passed" on a closed mesh.
- **Choice:** rebuild the cache and rewrite the probe per test. Measured cost: 38 s for all 11 tests
  at two periods, which is not worth optimising.
- **Ruled out:** hoisting `mesh_device` to module scope. It would work, and it would diverge from
  every other device test in the package and from `SubmeshPool`'s quiesce discipline (`DEC-077`).
- **Evidence:** the failing run (2 failed, 9 passed) and the passing one
  (`raw/G-KV-TABLE_20260904T172909Z.log`, 11 passed).
- **Confidence:** high.
- **Falsifier:** none — it is a lifetime rule, not a judgement.
- **Revisit if:** the suite's device time becomes a problem.
- **Blast radius:** `G-KV-TABLE`. **A note for P9 and for any later phase:** never cache a device
  tensor across a function-scoped `mesh_device`.

---

### DEC-106 — The serving gates run at chunk **256** / capacity **2816**, so `G-MOCK-MIG` is comparable to `G-MESH-KV`
- **Phase / module:** P10 / `G-REQUEST`, `G-MOCK-MIG`
- **Date (UTC):** 2026-09-04
- **Trigger:** the gate geometry is a free choice, and `BRINGUP_RECIPE.md:1956-1961` makes a specific
  demand of it: `G-MOCK-MIG` "is the strongest evidence in the whole bring-up, because it is a
  second, device-less reader in a different process agreeing with the on-device `G-MESH-KV` number
  **at the same shape**. Compare the two explicitly."
- **Constraints, all binding at once:**
  `CHUNK % (SP*32) == 0`; `MAX_SEQ_LEN % CHUNK == 0`; `MAX_SEQ_LEN >= chunks * CHUNK`;
  `MAX_SEQ_LEN > CHUNK` **strictly** (at equality the SP bootstrap core runs instead of the ring —
  "anything you measure is measuring the wrong path", `BRINGUP_RECIPE.md:1948-1951`); and the PCC arm
  must not read past the golden trace's **1024** tokens.
- **Choice:** `PREFILL_CHUNK_SIZE=256`, `PREFILL_MAX_SEQ_LEN=2816`, `PREFILL_NUM_USERS=1`;
  11 producer chunks for `G-REQUEST` (2816 tokens, the doc's own chunk count) and **4** for
  `G-MOCK-MIG` (4 x 256 = 1024 = exactly the golden's length).
- **Why 256 and not 512:** `G-MESH-KV` measured **both** chunked arms, and its `chunk 256`,
  4-chunk, 1024-token row is the one this reproduces token-for-token — same tokens, same golden, same
  number of `prefill_chunk` calls, same `sp_ring` core. The recipe asks for a comparison; this makes
  it an identity rather than an analogy. (256 % 128 = 0, 2816 / 256 = 11, 2816 > 256.)
- **Why `num_users=1` and not the doc's 2:** with one prompt every slot's KV is byte-identical, so a
  second user adds no discriminating power (`PREFILL_MIGRATION_TESTING.md:301-304`) — it would only
  double the cache and the read-back. What *would* discriminate is two **different** prompts via
  `PREFILL_PRODUCER_SLOT_TRACES`, and that needs a second golden trace: `R-044`.
- **Why the deployment pair is run separately:** the manifest pins the real deployment geometry
  (8192 / 131072), which no golden can score, so `G-REQUEST` gets a second arm at those values with
  no PCC — which is also what closes half of `R-039`.
- **Evidence:** `06_GATES.md` `G-MESH-KV` (chunked, chunk 256: min K 0.9967844 / V 0.9866232);
  `G-MOCK-MIG`'s own row.
- **Confidence:** high.
- **Falsifier:** a `G-MOCK-MIG` number materially different from `G-MESH-KV`'s at the same shape —
  which would mean the two readers disagree, and that is the whole point of running it.
- **Revisit if:** a deeper golden trace is generated.
- **Blast radius:** `G-REQUEST`, `G-MOCK-MIG`.

---

### DEC-107 — `kv_migration_stages` is deliberately **absent**, and `kv_migration_base_address` returns K's
- **Phase / module:** P10 / `tt/tt_prefill_runtime.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** this model migrates **two** tensors (K and V), and the doc says the multi-cache hook
  is for exactly that case: "Implement it instead of `kv_migration_base_address` when your model
  migrates SEVERAL caches" (`ADDING_A_PREFILL_MODEL.md:158-162`).
- **Question:** implement `kv_migration_stages`, or the single-base hook?
- **What the engine does with each:** it selects the branch on `hasattr(runtime, "kv_migration_stages")`
  (`prefill_runner.py:613`). With the hook present it gathers **one layout per stage** and passes them
  back as `stage_layouts` (plural, `:634`); without it, it wraps the single base in one `KvCacheStage`
  (`:617`) and passes `stage_layout` (singular).
- **Choice:** do **not** define `kv_migration_stages`; define `kv_migration_base_address`, returning
  `int(kv_cache.k.buffer_address())`. Asserted absent by
  `G-RUNTIME::test_kv_migration_stages_is_deliberately_absent`, so it cannot reappear by accident.
- **Why:** the multi-stage path exists to *merge* per-stage layouts, and the merge is the same
  unimplemented multi-rank code the table builder refuses (`R-032`). Defining the hook would move
  this runtime onto that path and hand it arguments it cannot honour — the recipe's "must raise
  rather than silently discarding" applied one level up. And the table does not need it: each config
  reads its own tensor's `buffer_address()`
  (`models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:138`), so V's base reaches the table
  directly and the anchor is only the engine's own bookkeeping.
- **What it costs:** on the real-migration path the engine gathers a stage layout describing only K.
  For a single rank that layout is used for nothing this package reads, but it is unverified —
  `G-LOOPBACK` is where it would show, and that gate is scoped out (`DEC-103`, `R-043`).
- **Evidence:** `prefill_runner.py:613-623`, `:634`; the template makes the same choice with the same
  two-tensor cache (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:370-373`).
- **Confidence:** medium-high — the reasoning is solid, the real path is unrun.
- **Falsifier:** a loopback run in which the engine's stage bookkeeping needs V's base.
- **Revisit if:** `G-LOOPBACK` runs, or multi-rank is implemented.
- **Blast radius:** `tt/tt_prefill_runtime.py`; `G-LOOPBACK` (unrun), `R-043`.

---

### DEC-108 — `metadata_msg` must be **accepted and ignored**, not refused — and `G-RUNTIME` cannot prove that
- **Phase / module:** P10 / `tt/tt_prefill_runtime.py`, `tests/unit/test_prefill_runtime_chunked.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the **first served chunk** of the first `G-REQUEST` attempt. The runner opened the
  mesh, loaded 15 GB of weights, compiled, entered the request loop, logged
  `CHUNK_START c=0 ... slot=0 [0,256)` — and died:

  ```
  File ".../tt_prefill_runtime.py", line 420, in prefill_chunk
      raise NotImplementedError(
  NotImplementedError: metadata_msg is the engine's trace-safe metadata tensor and needs
  config.use_trace plus capture_trace, neither of which this runtime implements (risk R-024).
  ```

- **What was wrong.** P7 wrote that refusal believing `metadata_msg` belonged to the trace path, and
  P7's own `G-RUNTIME` audit **passed it clean**. It is not a trace object at all: the engine reads
  it straight out of the H2D socket (`prefill_runner.py:156-159`), decodes it into the very
  `slot_id` / `actual_start` / `actual_end` it passes alongside (`:142-144`), and hands the raw
  tensor down so that a **pipeline** rank can forward it verbatim over D2D (`:304-312`). On a
  single, last rank nothing further happens to it. So in request mode it is **never `None`**, and a
  runtime that refuses a non-`None` value cannot serve a single chunk.
- **Choice:** accept and ignore it, exactly as `request_id` already was — `del request_id, metadata_msg` —
  and do **not** deallocate it, because the engine owns it (it frees it on the shutdown sentinel,
  `:359`). `d2h_service`'s refusal stays: that one really is `None` unless
  `PREFILL_LAYER_ACK_D2H=1` (`:707`, `:730-736`).
- **The method lesson, which is bigger than the bug.** `G-RUNTIME` is a **static** audit: it proves
  the signature *binds* the engine's call. It cannot know which of the values the engine binds are
  `None` in practice, so it cannot distinguish "accepted" from "accepted and then rejected at
  runtime". A refusal placed on a parameter the engine always populates is exactly as fatal as a
  missing parameter, and costs the same mesh open and weight load to discover — the failure the gate
  was written to prevent, reached from the other side.
- **What was added so it cannot recur:** two tests.
  `test_prefill_chunk_accepts_the_engines_always_present_metadata_msg` drives a call carrying a
  non-`None` `metadata_msg` **past** that point to the next refusal in line, proving it was consumed;
  and `test_every_parameter_the_engine_always_passes_is_accepted_or_used` reads the engine's keyword
  set out of the AST walk and asserts the runtime's body contains no `<param> is not None` refusal
  for any of them **except** `d2h_service`, with the reason that exception is legal stated in the
  assertion message.
- **Evidence:** the failing runner log (`raw/G-REQUEST-runner_20260904T173012Z.log`, retained: it is
  the evidence for this entry); `prefill_runner.py:142-159`, `:286-295`, `:304-312`, `:359`;
  `raw/G-RUNTIME_20260904T173636Z.log` (72 passed with the two new tests).
- **Confidence:** high.
- **Falsifier:** a future engine that passes `metadata_msg=None` on some path — harmless here, since
  the parameter is ignored either way.
- **Revisit if:** this runtime ever implements the trace path, where `trace_metadata_msg` becomes the
  forwarded object instead.
- **Blast radius:** `tt/tt_prefill_runtime.py::prefill_chunk`, `G-RUNTIME`, `G-REQUEST`,
  `G-MOCK-MIG` — and the recipe's description of what `G-RUNTIME` proves.

---

### DEC-109 — The two-terminal gates are driven by **one script**, not two terminals
- **Phase / module:** P10 / `G-REQUEST`, `G-MOCK-MIG`
- **Date (UTC):** 2026-09-04
- **Trigger:** `BRINGUP_RECIPE.md:1940-1952` specifies these gates as a two-terminal recipe, and this
  session has no two terminals.
- **Question:** how are the runner and the producer sequenced, given that the producer must not
  connect before the runner has exported its H2D descriptor, and the runner must be waited on
  afterwards?
- **Options considered:**
  1. Start both and hope. The producer's `H2DStreamService.connect` has a 60 s timeout
     (`PREFILL_H2D_CONNECT_TIMEOUT`), so it would *usually* work — the runner needs ~90 s for the
     mesh open, the 15 GB weight load and `compile()`, so it would usually **not**. A flaky gate is
     worse than a slow one.
  2. Sleep a fixed interval. Encodes today's weight-load time as a constant.
  3. Poll the runner's log for `setup complete, entering request loop`, then run the producer in the
     foreground, then `wait` on the runner and fail on either exit code.
- **Choice:** option 3, as a shell script in the session scratchpad (**not** in the package: the P3
  tree contracts 41 files and this is neither a deliverable nor something P9 should have to audit).
  Both processes' stdout is `tee`d to `bringup_log/raw/<GATE>-{runner,producer}_<stamp>.log`, so the
  evidence is exactly what two terminals would have produced.
- **Why it is better than two terminals, not merely equivalent:** the barrier is a real
  precondition rather than an operator's judgement, and it is itself a control — a runner that dies
  during the weight load fails the gate as "never entered the loop" instead of as a producer connect
  timeout, which is a different bug. The script also fails on the **runner's** exit code, which a
  human watching two terminals routinely forgets: the producer exits 0 quite happily while the
  runner is dying behind it, and `G-REQUEST`'s whole claim includes the runner's clean shutdown.
- **What it costs:** the exact commands live in the ledger and in `08_PREFILL_INTEGRATION.md` §2
  rather than in a runnable file in the tree, so reproducing the gates means reading them off the
  ledger. Stated for P9.
- **Evidence:** the four `raw/G-REQUEST*` and two `raw/G-MOCK-MIG*` logs, each pair carrying the same
  timestamp; the `producer_rc=0 runner_rc=0` line the script prints.
- **Confidence:** high.
- **Falsifier:** a gate that needs three processes (Gate 2 does — endpoint, runner, driver), where
  this shape would need extending rather than reusing.
- **Revisit if:** `G-LOOPBACK` comes into scope.
- **Blast radius:** `G-REQUEST`, `G-MOCK-MIG`; no package file.

---

### DEC-110 — One log string was reworded **after** its gate ran, and the transcript keeps the old text
- **Phase / module:** P10 / `tt/runners/adapters/llama.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** `DEC-098`'s measurement. `build_runtime` logged
  `loading real bf16 weights from ... (safetensors read) ...`, and the call it announces takes
  **46 ms** and — with the weight cache populated — reads no bytes at all, because `safetensors`
  memory-maps and nothing faults the pages in. A reader debugging a slow start-up would look in
  exactly the wrong place.
- **Question:** reword it (and diverge from the gate transcripts), or leave a misleading message
  standing so the evidence matches the code byte-for-byte?
- **Choice:** reword it to `mapping the bf16 checkpoint at ... (safetensors, lazy)`, and record the
  divergence here and in `08_PREFILL_INTEGRATION.md` §5 rather than editing the transcript.
- **Why:** raw logs record what happened; rewriting one to match new code would make the evidence
  less trustworthy, which is `BRINGUP_RECIPE.md` §0.2 rule 4 ("Raw logs from before a rename keep
  the old path. That is correct, not stale"). And nothing `G-REQUEST` or `G-MOCK-MIG` asserts depends
  on the string — their claims are chunk counts, exit codes and PCCs.
- **Scope note, stated rather than left for a reader to infer from timestamps:** the change
  post-dates `raw/G-REQUEST-*`, `raw/G-MOCK-MIG-*` and `raw/G-KV-TABLE_*`. It is a `logger.info`
  f-string with no other effect, and `P10-REGRESSION` ran after it. This is the same shape of note
  `DEC-093` made in P8 for a one-line signature change, and it is made for the same reason.
- **Confidence:** high.
- **Blast radius:** one log line; no gate's assertion.

---

### DEC-111 — `stage_layout` is a **list of one dict per rank**, and the first guard got it wrong twice
- **Phase / module:** P10 / `tt/runners/kv_chunk_table.py`, `tests/unit/test_prefill_runtime_chunked.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** an independent read of the new code against the engine's call sites, run *after*
  `G-ADAPTER`, `G-REQUEST`, `G-MOCK-MIG`, `G-KV-TABLE` and `G-RUNTIME` had all passed. Both defects
  below survived every one of them.
- **Defect 1 — the type was wrong, and it would have blocked every real migration run.**
  `assert_single_rank_stage` required `stage_layout` to be a **dict** and raised `TypeError`
  otherwise. It is a **list**: `allgather_kv_stage_layout` builds `stages = []` and appends one dict
  per rank in `for rk in range(size)`
  (`models/demos/common/prefill/runners/migration.py:315-334`), `allgather_kv_stage_layouts` returns
  one such list per migratable stage (`:287-291`), and the engine passes `stage_layouts[0]` — stage
  0's **per-rank list** — as `stage_layout` (`prefill_runner.py:634`). Every other reader in the tree
  iterates it: `models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py:929` tests single-rank with
  `all(len(layout) == 1 for layout in stage_layouts)` and
  `models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py:394` sums `s["count"]` over it.
  So every `PREFILL_ENABLE_MIGRATION=1` run — real worker (`:674`), file export (`:655`) **and**
  mock-with-migration (`:644`) — would have died with `TypeError` on a perfectly valid single-rank
  configuration.
- **Defect 2 — the multi-rank protection it advertised did not exist.** Two of the guards compared a
  value with itself. `num_layers` is `config.num_layers`, which **is** `params.num_layers`, which
  **is** this rank's `num_my_layers` (`prefill_runner.py:463`, `:481`) — and the engine passes that
  same `num_my_layers` at `:647`. So `num_my_layers != num_layers` can never fire, and neither can
  `stage_layout["count"] != num_layers`. The `first_layer_idx != 0` guard cannot fire either,
  because only rank 0 builds the table (`:643`, `:654`, `:673`) and rank 0's first layer is 0. On a
  4-rank pipeline all three guards would have passed and the builder would have emitted a table
  declaring `num_layers = 8` for a 32-layer model. `R-032`'s stated mitigation was vacuous.
- **Why no gate caught either.** `G-MOCK-MIG` ran `PREFILL_MOCK_MIGRATION=1` **without**
  `PREFILL_ENABLE_MIGRATION=1`, which takes `prefill_runner.py:570` / `:699` — the only two call
  sites that pass **no** `stage_layout` at all, so the guard returned early at its `None` check.
  `G-RUNTIME` asserted the wrong contract in two of its own tests: it *required* a `TypeError` for a
  list and *accepted* a bare dict as "the single-rank shapes the engine really passes". A test
  written from the same wrong belief as the code cannot falsify it.
- **Choice — the fix, and it is the same argument in both halves:** the gathered **list** is both
  the real type and the only argument that carries information from other ranks. So the guard now
  takes a sequence, refuses a bare `dict` naming this decision, refuses an empty one, and refuses
  `len(stage_layout) != 1` — which *is* the multi-rank signal — before checking that the single
  stage spans `[0, num_layers)`.
- **And a gate arm that can see it:** `PREFILL_ENABLE_MIGRATION=1` **with**
  `PREFILL_MOCK_MIGRATION=1` takes `prefill_runner.py:626` (the real `allgather_kv_stage_layouts`)
  and `:644` (the build with `stage_layout=stage_layouts[0]`) and needs **no** tt-llm-engine
  binaries, because the worker handshake is only in the `else` branch at `:673`. That is
  `G-MOCK-MIG`'s second arm, and it also exercises `kv_migration_base_address` for the first time —
  closing two items `R-043` had listed as unproven.
- **Evidence:** `migration.py:287-291`, `:315-334`; `prefill_runner.py:613-634`, `:644-650`;
  `models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py:929`;
  `models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py:394`; `G-MOCK-MIG` arm 2's log.
- **Confidence:** high — the type is read out of the producing function, and a new `G-RUNTIME` test
  (`test_the_gathered_stage_layout_really_is_a_list_of_dicts`) now AST-checks that function so the
  belief cannot drift again silently.
- **Falsifier:** the engine changing `allgather_kv_stage_layout` to return a dict — which that new
  test would report.
- **Revisit if:** multi-rank is implemented; the `len == 1` check is where the merge goes.
- **Blast radius:** `tt/runners/kv_chunk_table.py`, `G-RUNTIME`, `G-MOCK-MIG`; and `R-032`, whose
  claimed mitigation was not real until now.
- **The method point, and it is the same one as `DEC-108`.** Both defects are refusals written from
  a parameter's **name** rather than from what the engine actually puts in it, and in both cases the
  static audit passed and only a real run could tell. `DEC-108` was found by a served chunk;
  this one by a second reader, because the run that would have found it is a *branch* no gate had
  taken. **A refusal is code, and it needs its own positive test on the engine's real value** — an
  assertion that a bad input raises is only half of it.

---

### DEC-112 — `build_kv_chunk_table` refuses more than one supported chunk size
- **Phase / module:** P10 / `tt/tt_prefill_runtime.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the same review. `TtPrefillRuntimeConfig` supports `additional_chunk_sizes`,
  `prefill_chunk` takes a `chunk_size` override, and `config.max_chunk_size` is `chunk_sizes[0]` —
  the *largest*, which need not be `config.chunk_size`.
- **Question:** the table describes **one** block-cyclic period. What if the cache was written at
  two?
- **What goes wrong:** nothing crashes. A cache whose rows were laid out by two different periods
  has no single position -> address map, and `build_kv_chunk_table` would emit a table for
  `config.chunk_size` — addresses that resolve, decode as plausible KV, and are wrong. It is the
  only silent-wrong-answer this module can produce, and `G-KV-TABLE` cannot see it because that gate
  writes each period into its own fresh cache.
- **Choice:** refuse when `len(config.chunk_sizes) != 1`.
- **Why not support it:** a correct answer needs one table per period and a reader that knows which
  period each position belongs to. That is real work with no caller: the engine never passes
  `chunk_size` to `prefill_chunk` at all (`prefill_runner.py:287-296`) and the adapter never sets
  `additional_chunk_sizes`, so the multi-size path exists only for this package's own harnesses.
- **Evidence:** `G-RUNTIME::test_build_kv_chunk_table_refuses_more_than_one_block_cyclic_period`.
- **Confidence:** high.
- **Falsifier:** a deployment that serves two chunk sizes into one cache and migrates it.
- **Blast radius:** `tt/tt_prefill_runtime.py::build_kv_chunk_table`; `G-RUNTIME`.

---

### DEC-113 — `load_hf_config` takes the bundled default with **no path arithmetic**
- **Phase / module:** P10 / `tt/runners/adapters/llama.py`
- **Date (UTC):** 2026-09-04
- **Trigger:** the same review. `hf_model_default` is a **repo-relative** string (as the template's
  is, and as the engine prints it — `prefill_runner.py:375`), and the first version compared
  `os.path.abspath(config_dir + "/config.json")` against the absolute `BUNDLED_CONFIG_PATH`.
- **What goes wrong:** `abspath` resolves against the process **CWD**. Started from anywhere but the
  repo root the comparison fails even with `PREFILL_HF_MODEL` unset, the code takes the
  foreign-checkpoint branch, `os.path.isfile` fails, and it raises
  `FileNotFoundError: PREFILL_HF_MODEL='models/demos/…' has no config.json` — naming an env var the
  operator never set. Every gate in this phase ran from `$TT_METAL_HOME`, so none of them saw it.
- **Choice:** when `PREFILL_HF_MODEL` is unset, return the bundled config **directly**, with no path
  joined, no `abspath` and no comparison. Only an explicitly-set `PREFILL_HF_MODEL` reaches the
  comparison, and that one is compared with `os.path.realpath` on both sides.
- **Why this shape:** the default case had no need of path arithmetic in the first place —
  `ModelArgs.load_bundled_config()` already knows where the file is (`tt/model_config.py:53`, built
  from `_PKG_ROOT`). Deleting the arithmetic is better than making it CWD-independent.
- **Evidence:** `G-ADAPTER` (unchanged: it exercises both branches, and
  `test_load_hf_config_refuses_a_disagreeing_checkpoint_config` passes an absolute `tmp_path`).
- **Confidence:** high.
- **Note for P9:** `G-ADAPTER::test_identity_and_default_paths_are_set` still does
  `os.path.isdir(adapter.hf_model_default)`, which passes only from the repo root — the same CWD
  dependence, now only in a test. Left as-is because pytest is always run from the root here and
  P9 item 1 runs from there too; flagged rather than hidden.
- **Blast radius:** `tt/runners/adapters/llama.py::load_hf_config`.

---

### DEC-114 — Every recipe citation in the package was re-pointed after an out-of-band kit edit
- **Phase / module:** P10 / `scripts/verify_citations.py` and 26 files across the package
- **Date (UTC):** 2026-09-04
- **Trigger:** the final `G-CITE` pass went from clean to **38 mismatched** with no change to any
  package file. `BRINGUP_RECIPE.md` had grown 23 lines while this session was live — a new passage
  after `:1599` about the very defects `DEC-108` and `DEC-111` record — and every reference past that
  point shifted by exactly +23.
- **Options considered:**
  1. **Leave them.** Pass 1 stays red, Appendix C item 7 fails, and the phase cannot be recorded
     clean. Not viable.
  2. **Re-point only this phase's own refs.** Leaves P0-P8's prose refs wrong. They are *in range*,
     so pass 2 reports them `resolved` — which is exactly the failure `R-016` describes and `R-042`
     was opened for.
  3. **Re-point every reference in the package**, mechanically, by the measured offset.
- **Choice:** option 3. `BRINGUP_RECIPE.md:N` with `N >= 1600` -> `N + 23`, in `CITES` (**104** rows)
  and in prose (**26** files), including the `N-M` range form. **Raw logs excluded**: they record
  what ran, and rewriting one would make the evidence less trustworthy (`BRINGUP_RECIPE.md` §0.2
  rule 4).
- **Why mechanical rather than by hand:** the offset is a measured constant, and every one of the 38
  content-checked failures reported the exact line its needle had moved to, so the constant is
  confirmed 38 times over. A hand pass over 26 files is where the "+209 offset" error `§1.6` records
  came from.
- **What this touched that this phase does not own:** `README.md`, `03_OUTLINE.md`,
  `04_CCL_PLAN.md` and fourteen P5-P8 test files. Their content is unchanged — only the line number
  each reference points at. No `tt/` module was affected; the one `tt/` file this phase modified is
  `tt_prefill_runtime.py`, for the migration hooks it owns.
- **Evidence:** `raw/G-CITE_20260904T184140Z.log` (604/604, 1169/1169, 128/128); the 38 failures and
  their reported target lines, all +23; `git diff models/demos/common/bringup/` (the 23-line
  addition and its insertion point).
- **Confidence:** high — the offset is verified by the verifier itself, 38 times.
- **Falsifier:** a reference whose needle moved by something other than +23, which pass 1 would
  still report.
- **Revisit if:** the recipe is edited again. `R-017` now records two instances.
- **Blast radius:** every `BRINGUP_RECIPE.md` reference in the package; no behaviour.
