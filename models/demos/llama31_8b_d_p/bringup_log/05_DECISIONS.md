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
  `BRINGUP_RECIPE.md:1811` (the "too good" symptom when `max_seq_len == chunk_size`).
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
  (`BRINGUP_RECIPE.md:1732`, `:1740`, `:1746`); `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:56`
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
  switch, not taken); Appendix A `G-MLP` "≥ 0.999 @bf8_b, ≥ 0.9995 @bf16" (`BRINGUP_RECIPE.md:1730`).
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
     `TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_` prefixes anyway (`BRINGUP_RECIPE.md:1808`), so a
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
  `BRINGUP_RECIPE.md:1685-1687` because it logs and must never break a run);
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
