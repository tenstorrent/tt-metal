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
