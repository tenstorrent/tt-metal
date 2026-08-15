# Findings — running `tt_hw_planner` / `perf_automation` on Voxtral-TTS Block 1

Running log of every issue hit while driving PR #46283's pipeline end-to-end, and what was done
about it. Written so each entry stands alone: symptom first (what a user actually sees), then the
real cause, then the fix, then how it was verified.

**Why this file exists.** The failures so far have not announced themselves — each one surfaced
three layers downstream of its cause, wearing a diagnosis that pointed at the wrong thing. A log
of "what it said" next to "what it was" is the useful artifact.

## The experiment

Port Voxtral-TTS **Block 1** (the 3.4B autoregressive backbone) with the tool, blind, and compare
against a hand-built TTNN implementation of the same block that has been through 74 recorded
optimization experiments.

| | |
|---|---|
| model given to the tool | `/localdev/lserbedzija/hf_models/voxtral-tts-backbone` |
| what it is | Block 1 exported to HF format — stock `MistralForCausalLM`, 26 layers, dim 3072, GQA 32/8, head_dim 128, SwiGLU 9216, RoPE θ=1e6 |
| provenance | `scripts/export_backbone_hf.py` from the hand-port's own history, recovered at `a4b9382c96` |
| verified | PCC **1.00000298** vs the fp32 CPU reference, max\|diff\| 2.6e-04 on rms 2.59 |
| contains | weights + `config.json` only — **no TTNN code, no tuning, no findings** |
| blindness | run from a worktree of `pr46283`, which has no `voxtral_tts` directory at all |
| hardware | one Blackhole p150b, `--box P150 --mesh 1,1` |
| target to beat | **~15.9 ms per decode step**, batch 1 (the hand-port's Block 1, measured device-bound: eager 15.907 / traced 15.922) |

---

## RESULT SO FAR — the port is CORRECT; the tool could not tell

Bring-up **succeeded**. Verified by running the generated tests directly on the P150, not by
reading the tool's status files:

```
[bringup] achieved PCC=0.9998358  target=0.99  component=attention
[bringup] achieved PCC=0.9999971  target=0.99  component=decoder_layer
[bringup] achieved PCC=0.9999944  target=0.99  component=m_l_p
[bringup] achieved PCC=0.9999874  target=0.99  component=r_m_s_norm
[bringup] achieved PCC=0.9999992  target=0.99  component=rotary_embedding
                                                        pytest exit=0, 5/5
```

All five components were finished by **07:22 on 2026-08-13**, about 17 minutes into the run. The
loop then ran a further ~1.5 h re-doing completed work, reporting `graduated 0`, and would not
have stopped on its own. See F6: the agent had no way to report what it had done.

---

## ★ FOR THE HAND-PORT — what to actually do, ranked

The point of the experiment. Everything below is derived from a tool that never saw the hand-port
(see the blindness audit). Full detail in the O4* entries; this is the short list.

### Take these

| # | change | evidence | risk |
|---|---|---|---|
| **1** | **Rotate Q+K in ONE call** — slice q+k out together, split heads once into a 40-head tensor, rotate once, slice apart. NOT ttnn's fused q+k rope operator (§6.23 rejected that correctly, for the interleaved convention); the *same* `rotary_embedding_hf` on a wider tensor. | exact arithmetic, PCC bit-identical, −1.99% on the identical model. 52 rotary launches/token → 26; `[gpt-24]` priced a comparable 26-launch saving at 0.405 ms | **check first:** decode mode wants cos/sin sharded to match, and §6.44 documents the trap (*"RoPE on a core whose cos/sin table lives elsewhere returns 3.4e38"*) |
| **2** | **Give `_PRG_W2` its own grid and re-sweep it.** One `_MM_GRID` for all five decode matmuls cannot be right for both K-light projections and the deepest K-reduction in the model. | measured curve on the identical op: `96c 0.1589 / 48c 0.0960 / 32c 0.1137 / **24c 0.0934** / 16c 0.1084`, a 1.70× spread. `_PRG_W2` sits at 72c. | none — precision-neutral. Sweep and keep what wins |
| **3** | **Audit the decode path for the batch-1 / seq-1 pathology** (see ★ THE PATTERN). Four instances in one model. | argmax 32× the bytes; rotation 160 tiles where 4 had data; head creation collapsed to one core; a join moving 8 MB where 64 KB was real | none — these are pure waste |
| **4** | **Keep the decode residual stream in ONE shard, the whole way down.** `_norm` returns `sharded_norm(..., _L1)` — it shards internally, converts back to **L1 interleaved** on the way out, and `ttnn.linear` converts straight back in. The tool went further than the one handoff: both norms in a block and the next layer's are built on the same dim, so the stream *never has to leave the shard* across all 26 layers. | 0.80 ms of layout ops removed on the QKV handoff alone (`sharded→interleaved` 1152→896 calls, `interleaved→sharded` 1024→768), then a further −0.11 ms/token carrying it through the residual adds | **the catch:** the chain only exists where the grids agree. Norm is `(8,4)`=32, `_MM_GRID` is `(12,6)`=72. The tool paid 3.72→3.98 ms moving its norm 32→48 to meet the projection. Do this **together with #2** — pick `wqkv`'s re-swept grid to be one the norm can share. Note `[gpt-27]` (residual as matmul bias) already removes the *add* launches; this is about the *layout* round-trip, and the two compose |

### Re-open these, don't assume they're settled

| # | what | why |
|---|---|---|
| **4** | **§6.8 — device argmax rejected in favour of host.** | The A/B was scored against a **single-core** kernel that didn't have to be single-core. `ttnn.argmax` picks its path from input LAYOUT; a TILE input single-cores the scan *and* pads the row 32×. Re-run with `to_layout(ROW_MAJOR)` in front. Host may still win on a 33 KB reduce that already ends in a D→H copy — but the number you have doesn't answer that. |
| **5** | **§6.44 — fused K/V cache write deleted as 0.687 ms/step slower.** | The tool measures it **faster** on the same board: 0.402 → 0.210 ms/token, moving V one core exactly as `[gpt-24]` did. The layouts are materially similar — you already call `nlp_create_qkv_heads_decode` on a sharded operand — so I **cannot** explain the disagreement, and my first attempt to (a layout you hadn't adopted) was wrong. An unexplained 0.19 ms/token swing is still worth one A/B. |
| **6** | **`_MM_GRID` generally.** | Tuned at §6.52, then §6.65/§6.67/§6.72 changed the structure around it. A structural change invalidates earlier knobs silently (O4m: a stale cap cost 15% on one op). |

### Explicitly do NOT take

- **`down_proj`/`w2` → bf8_b.** This is §6.16, which you measured and declined. The tool takes it only because a 0.95 gate has no reason not to. Your call stands.
- **DRAM-sharding the LM head weight.** Real bandwidth insight (interleaved buffers round-robin across all 8 banks; the head ran at 256 GB/s where projections manage 340–360), but it costs a **second 226 MB copy**, and Block 1's LM head isn't on your critical path.

### What the experiment confirms about work you already did

Six independent agreements, each reached without sight of your code: the width-sharded decode norm
(§6.67), the traced decode loop (§6.65), decode matmul program configs (§6.52), the fused `wqkv`,
the hand-rolled head split over the fused op (§6.72 — *"dispatches fell 3413 → 2867 yet it got
slower; these were view ops doing no work"*), 2 cores/head on the decode SDPA (`[gpt-21]`), and that
`activation=` never fuses while `fused_activation` does (`[gpt-26]`).

**And one thing the tool structurally cannot check:** `[gpt-21]` records SDPA settings that were
faster but *"NOT SAFE — position sweep"*. The tool gates on PCC at a single length. Nothing in it
would have caught that.

---

## OWNERSHIP — who should fix what

Split by owner so the PR feedback can be lifted straight out of this file. **Only the first table
is for the PR author.** The second is our own setup, recorded so it is never mistaken for a tool
defect.

### A. TOOL DEFECTS — for the PR author

| # | one line | fix? | effort |
|---|---|---|---|
| **F6** | `mcp` is **declared nowhere** → all 10 agent tools silently absent → 11 h stall | **YES — first** | trivial |
| **F2** | the READY verdict can never fire (`lambda _: []`); best compat report = surest failure | **YES** | one line |
| **F1** | a local model dir gets a reduced probe → refused 3 stages later, wrong diagnosis | **YES** | small |
| **F7** | one reporting channel, no cross-check against disk, unbounded retries | **YES — highest value** | medium |
| **F3** | a Python exception reported as "the PCC gate rejected the output" | **YES** | small |
| **F9** | a local model dir is mistaken for a demo dir (`optimize` unusable by model id) | **YES** | small |
| **F8** | the in-place refusal returns `rc=1`, so the supervisor resets the card 3x | **YES** | one line |
| **F10** | the F9 workaround loses the model id → `optimize` cannot build its own PCC gate, though it wrote one | **YES** | small |
| **F5** | systemic-pattern detector counts error NAMES, not families | **YES — low prio** | small |
| **F4** | deliberately-wrong constructor, repaired on retry | **NO — works as designed** | — |
| **F11** | documented `--max-rounds` default is 20; the real one is 3, and it is the ONLY exit | **YES** | one line |
| **F12** | the fusion rung reaches for a grid where it should reach for a program config | **YES** | medium |
| **F13** | generated stubs swallow fast-path exceptions, so a perf regression passes the PCC gate | **YES** | small |
| **F14** | "producer emits the consumer's shard" must check the consumer's PROGRAM CONFIG grid | **YES** | medium |
| **F15** | `plan` and `compat` disagree about what the model IS | **YES** | small |
| **F16** | the block table degrades to EMPTY, which reads identically to "nothing needed" | **YES** | small |
| **F17** | machine-readable structure is declared and never read | **YES** | small |
| **F18** | the architecture gate tests the model's NAME, not its structure | **YES — fix tested** | small |
| **F19** | template dispatch silently runs a DIFFERENT model; the template can be the tool's own prior output | **YES** | medium |
| **F20** | ⚠ REVISED — the meta-plan is wired to stdout, not control flow; on this run the pipeline ignored it and was RIGHT | **partial** | — |
| **F21** | `trust_remote_code` is a ONE-MODEL allowlist; the two halves of the pipeline disagree | **YES** | small |
| **F22** | the isolation worktree silently ignores uncommitted edits to the tool's own source | **YES** | small |
| **F23** | ⚠ CORRECTED — capture drivers guess where the config already says they should not; 3 of the 4 misses were OURS (S5) | **partial** | small |
| **F25** | decomposition children lose the parent path prefix, and the plan is copied from another model | **YES** | small |
| **F26** | report what the gate MEASURED, not what was collected (`captured 7/7` ≠ used) | **YES** | small |
| **F27** | the captured input is DISCARDED where one `deepcopy` would have kept it | **YES** | one line |
| **F28** | the entire end-to-end verdict rests on ONE prompt (n=1) | **YES** | small |
| **F29** | the CLI's 0.95 `--pcc-target` overrides the engine's documented 0.99 — the threshold SETS quality, it does not merely gate it | **YES — highest value of the three-block run** | one line |
| **F30** | the drift gate detects the stale template and is wired never to block (*"Never raises"*) | **YES** | small |
| **F31** | a profiled child that aborted (SIGBUS) is reported as a missing CSV | **YES** | small |
| **F32** | `termination_check()` blocks 30 min with no progress channel; the retry never returns | **YES** | medium |
| **F33** | `worktree-list` can never print ORPHAN (`id(s) in orphans`), so dead worktrees accumulate looking active; `PermissionError` is also misread as dead | **YES** | one line |
| **F34** | the overlay store silently restores a deleted model over a clean HEAD, so a from-scratch run is unreachable and two runs from one commit differ invisibly; `overlay-drop` also fails to empty its scope | **YES — reproducibility** | small |
| **F35** | backend selection is non-deterministic — identical runs picked different templates, the LLM ranker overriding its own top score, choosing between two entries whose paths are both missing | **YES — reproducibility** | small |
| **F36** | "PCC tests will use real inputs" is false — the graduation gate builds inputs with `torch.randn` and never loads the 43 MB captures or the captured `output.pt`; raising the threshold measures the wrong thing more precisely | **YES — the sharpest one** | small |

**If only one thing is taken: F6.** It is the difference between "this tool does not work" and
"this tool ported a 3.4B model correctly in ten minutes".

**If one thing is taken from the three-block run: F29** — a one-line default that this document
measures as the difference between e2e PCC 0.9586 and 0.9986 on the same code. F30 is its
structural twin: a gate the tool already has and declines to enforce.

### B. OUR SIDE — not tool defects, do NOT report these

| # | what | whose fault | note |
|---|---|---|---|
| **S1** | the HF export shipped no tokenizer | **ours** | our exporter predates this use; fixed by converting `tekken.json` (15/15 vs ground truth) |
| **S2** | `tt-perf-report` not installed → `optimize` preflight failed | **ours** | it **is** declared in `requirements-agent.txt`; we simply never ran that install. The tool detected it, refused cleanly, and named the fix — correct behaviour |
| **S3** | the model had to be converted to HF format at all | **ours / inherent** | Voxtral ships in Mistral-native format; every model this tool handles arrives as a `transformers` model. Not a defect |
| **S4** | six packaging defects, all in `transformers`, all hit in one afternoon | **ours / upstream** | see §S4; none of them the tool's |
| **S5** | the first HF wrapper exposed 26 empty `nn.Module()` placeholders | **ours** | caused 3 of the 4 capture misses originally written up as F23 |
| **S6** | our own `conftest.py` bootstrap shadowed the built `ttnn` inside the planner's scratch copy | **ours** | fixed; looks exactly like a tool defect because the tool creates the copy |
| **S7** | parking the Block-1 demo left its `family_backends` entry dangling | **ours** | this is what exposed F30 — and it left a local absolute path in a shared registry |

### C. CREDIT — things the tool got right

Worth saying, so the feedback is not one-sided:

- **It refuses to optimize while its own test suite is red** (`rc=3`, "a decision, not a crash"),
  and the message named the failing tests and the override flag. That gate did its job.
- **The supervisor distinguishes refusals from crashes** — a dedicated exit code, no retry, no
  device reset. F8 is one call site returning the wrong code, not a missing capability.
- **It caught a real silent-wrongness statically**: this config uses HF's newer `rope_parameters`
  while `tt_transformers` reads only `rope_scaling`, so scaling would be silently ignored at long
  context. Found before anything ran, with two concrete fixes offered.
- **It kept honest accounting**: the e2e report says "1/5 (20%) actually graduated (native stub)"
  and labels the other four `REUSE-wired`, rather than claiming credit for all five.
- **The port is correct.** 5/5 components at PCC 0.9998-0.99999, verified independently.

---

## F1 — a LOCAL model directory gets a degraded probe, and the run dies three stages later

**Status: FIXED** (`probe.py`) · severity: blocks any local-path model · reported: not yet

### What the user sees

```
Step 1/6  Static analysis (plan + compat)
  Summary: 11 ready  /  0 partial  /  0 missing
  Memory fit gate SKIPPED: no LLM-style memory model produced — typically a vision /
  multi-modal model whose memory budget is dominated by per-op scratch, not weights.

Step 2/6  Scaffold the demo folder
ERROR: unexpected compat verdict 'UNKNOWN'; refusing to scaffold
RUN ENDED: pre-flight/setup failed — model could not be loaded, scaffolded, or prepared
```

A model the tool had **just declared 100% supported** (11/11 components drop-in from
`tt_transformers`) is refused at the next stage, with an error naming neither the cause nor the
input that triggered it.

### The actual cause

`probe_model()` has two paths and only one of them does the work:

```python
def probe_model(model_id: str) -> ModelProbe:
    _validate_hf_id(model_id)
    if _is_local_model_dir(model_id):
        return _probe_local_model(model_id)     # returns early — no arch_spec, no memory_model
    ...                                          # Hub path: builds both
```

`_probe_local_model` reproduces the Hub path's category/dtype/param logic but stops before the
arch-spec and memory-model section. So a local directory returns `arch_spec=None`,
`config_status=None`, `memory_model=None` — while the Hub path returns all three.

### The cascade, which is the interesting part

Four layers, each degrading quietly and re-describing the problem as something else:

| layer | what it does with the missing memory model |
|---|---|
| `_probe_local_model` | returns `memory_model=None`. Says nothing. |
| memory-fit gate (`cli.py:1330`) | reports *"typically a vision / multi-modal model"* — **a guess, contradicted by the probe's own `category='LLM'`** |
| same gate | returns `("unknown", …)`; its own docstring says **"Caller SHOULD proceed."** |
| `scaffold.py:214` | treats `UNKNOWN` as fatal — **the opposite of what the gate asked for** |

Nothing anywhere says "local paths take a reduced probe".

### Evidence it is not the model

The probe read the config perfectly — 32 keys, `hidden_size 3072`, `num_hidden_layers 26`,
`num_attention_heads 32`, `model_type mistral`, `head_dim 128`, `intermediate_size 9216` — and
counted `total_params 3429020008`, `weight_bytes 6858040016`. Both correct. `plan` had already
sized the model across nine boards and called it an LLM with **CONFIDENCE: HIGH**.

### Fix

Extract the Hub path's arch/memory block into `_attach_arch_and_memory(probe, cfg, total_params,
weight_bytes)` and call it from **both** paths. Deliberately one function, not a copy — two copies
is how the paths diverged.

### Verified

```
                 before        after
config_status    None          True
arch_family      None          dense
memory_model     None          DenseTransformerModel
arch_spec        None          layers 26, hidden 3072, heads 32, kv 8, head_dim 128
```

### Worth noting

The tool's own XTTS-v2 registry entry uses a **local path** as its canonical id
(`canonical_hf_id="/local/ttuser/apande/models/XTTS-v2-hf"`), so the only TTS family in the
registry points at exactly the input shape that fails here. `optimize`'s documented usage also
takes local directories.

### Still open, not fixed here

The gate/caller contract is still contradictory: the gate documents `unknown` as "caller SHOULD
proceed" and `scaffold.py` refuses. F1's fix removes the *trigger* for this model but leaves the
disagreement in place — any other source of `unknown` will reproduce it.

---

## F2 — the READY verdict can never fire: an all-green compat report is refused

**Status: FIXED** (`compatibility.py`) · severity: blocks any architecturally-ready model that
is not already demo-wired · reported: not yet

### What the user sees

Identical to F1's symptom, which is what made it confusing — fixing F1 did NOT clear it:

```
  Memory fit gate PASSED: mesh `1,1` on `P150` -> FITS (comfortable)
  Summary: 11 ready  /  0 partial  /  0 missing
  Overall verdict:   UNKNOWN
Step 2/6  Scaffold the demo folder
ERROR: unexpected compat verdict 'UNKNOWN'; refusing to scaffold
```

### The actual cause

`_aggregate_overall` leaves `report.overall` at its `"UNKNOWN"` default and overwrites it only if
a predicate in `_OVERALL_FROM_STATUSES` matches. The third entry is the unconditional catch-all:

```python
(
    lambda _: [],        # <-- an empty list. `if predicate(report):` is ALWAYS False.
    "READY",
    "All required blocks already exist in models/tt_transformers/...",
),
```

| compat result | verdict |
|---|---|
| any block MISSING | `BLOCKED` — fires correctly |
| any block PARTIAL | `FEASIBLE WITH WORK` — fires correctly |
| **everything READY** | no predicate matches → stays `UNKNOWN` → scaffold refuses |

**The better the compatibility result, the more certainly the run dies.** A perfect report is the
one input that cannot produce a verdict.

### Why it survived this long

`_aggregate_overall` returns `ALREADY SUPPORTED` early for anything in `SUPPORTED_HF_MODELS` or
found by discovery — which is most models the tool is pointed at. The dead branch is only reached
by a model that is architecturally ready but **not yet wired as a demo**, i.e. exactly the
new-model case the tool exists to serve.

### Fix

`lambda _: []` → `lambda _: True`, with a comment recording what it broke, since the next person
reading a bare `True` will wonder why it is not simply `else`.

### Verified

```
before:  Overall verdict: UNKNOWN     11 ready / 0 partial / 0 missing
after:   Overall verdict: READY       11 ready / 0 partial / 0 missing
```

### Note on F1

F1 and F2 produce the *same* error message from the *same* line. F1's fix was necessary (the
memory gate genuinely had no model) but not sufficient, and the identical symptom made it look
like the first fix had not worked. Two independent defects, one error string.

---

## F3 — a Python exception is reported as "the PCC gate rejected the output", and triggers hours of the wrong work

**Status: OPEN (tool)** · severity: sends the loop down its most expensive path for a non-numerical
failure · reported: not yet

### What the user sees

```
FAILED simple_text_demo.py::test_demo_text[...] -
  Exception: No fallback tokenizer found for base model: voxtral-tts-backbone
...
  ESCALATING on PCC fail  model=/localdev/.../voxtral-tts-backbone
  The ALREADY-SUPPORTED routing produced output the PCC gate rejected. Drafting a NEW
  backend via auto-onboard and re-invoking `up` so the scaffold + per-component iterate
  loop runs.
```

Nothing was measured. The test raised before producing a number, and an **environment** problem
(no tokenizer on disk) was classified as an **accuracy** problem ("the PCC gate rejected"). The
response is the most expensive path the tool has: draft a new backend, re-scaffold, and port every
component with LLM agents. On this run that consumed ~1.5 h and 39+ agent rounds before it was
diagnosed by hand.

### Why it matters

The distinction is cheap to make — a pytest ERROR/exception is not a PCC failure — and the two
call for opposite responses: an environment fault should be reported and fixed in seconds, an
accuracy fault genuinely warrants the port. Mapping the first onto the second converts a
one-line fix into hours of device and agent time, and the log tells the user the model's numbers
were wrong when they were never computed.

### Suggested fix

Classify the gate result before escalating: if the test errored (exception / collection failure /
missing dependency) rather than producing a PCC below threshold, surface the exception and stop.
`_cli_helpers/failure_classifier.py` and `error_patterns.py` already exist for this kind of
triage; the escalation path does not consult them.

---

## S1 — scaffolding gap (OURS, not the tool's): the HF export shipped no tokenizer

**Status: FIXED** (`hf_models/make_tekken_hf_tokenizer.py`)

`export_backbone_hf.py` emits `config.json` + weights only — its original consumer
(`tt_transformers`) sourced the tokenizer separately. A real HF model ships one, so this is our
gap, and it is what triggered F3.

**Fixed faithfully rather than substituted.** Tekken is tiktoken-shaped, so `tekken.json` converts
directly: vocab entries are `{rank, token_bytes(b64), token_str}`, ids are `rank + 1000` (ids
0–999 are the 1000 special tokens), and only the first 130072 ranks are in the released vocabulary
(1000 + 130072 = 131072 = the embedding width). Verified `len(tokenizer) == 131072`.

**Validated against ground truth: 15/15 fixture cases match `mistral_common` exactly** — the text
ids the tokenizer produces appear verbatim inside the recorded prompt ids, across 8 languages,
digits, a symbol run, emoji and literal tab/newline.

### The trap in it, worth recording

The first conversion was wrong in a way **no round-trip test would catch**: it kept only the best
split per token instead of every valid split, and never sorted merges globally by the merged
token's rank. The result decoded to **byte-identical text** while emitting **55 tokens where the
truth has 26** — a correct-looking tokenizer that silently doubles sequence length, i.e. doubles
prefill cost and changes every downstream measurement. Only comparing ids against a known-good
tokenizer catches it.

---

## F4 — the deliberately-wrong constructor: the tool emits a call it expects to fail, and the repair never came

**Status: root cause identified; F5 is the fix** · severity: no component can graduate

### What the user sees

Five components, all matching the PyTorch reference — and **not one graduates**:

```
component          best PCC        failure class      gate is 0.99
decoder_layer      0.99999594      SHAPE
r_m_s_norm         0.99998738      UNEXPECTED_KWARG
rotary_embedding   0.99996236      UNEXPECTED_KWARG
m_l_p              0.99981016      MISSING_KWARG
attention          0.99969689      MISSING_KWARG
```

The port is CORRECT. Every failure is the test wrapper failing to construct the tt_transformers
class, before any number is computed.

### The actual cause — and it is by design

`bringup_loop.py:1977` emits ONE hardcoded constructor call into every component's test:

```python
canonical = {canonical_import_target}(
    mesh_device=device, args=args, state_dict=..., layer_num=0, dtype=ttnn.bfloat16,
)
```

with a comment that states the problem plainly:

> The exact constructor signature varies per class (Attention takes 14 args, MLP takes 11,
> RMSNorm different, RotaryEmbedding different). **The LLM refines this call on PCC failure.**

So the strategy is deliberate: **emit a call known to be wrong, let it crash, repair on retry.**
That is defensible — the signatures genuinely vary — but it makes the repair loop load-bearing.
On this run the repair never happened: `attempts` is **1** for every component after 39+ rounds
and 9 hours.

**This is not a stale-API problem.** The tool is running from its own branch; `tt_transformers`
there requires `tt_ccl`, `weight_cache_path`, `transformation_mats`, `configuration`, and the
template passes none of them. It would fail identically on any checkout.

### CORRECTION (2026-08-13): the repair DID happen — through the agent's generic tools

The first write-up of F4 said "the repair never came". That is wrong, and the truth is more
interesting. `allowed_tools` grants the agent ten MCP tools **and six ordinary ones** — `Read`,
`Edit`, `Write`, `Bash`, `Grep`, `Glob`. The MCP half was dead (F6); the ordinary half was not.

So the agent read the real `__init__` signatures, repaired every constructor, ran the tests with
`Bash`, and — unable to call `record_result` — wrote the graduation snapshots **by hand** with
`Write`. The file timestamps show it working through the list one component at a time:

```
07:16:40  attention.py          + attention.py.last_good_native
07:18:21  decoder_layer.py      + decoder_layer.py.last_good_native
07:19:37  m_l_p.py              + m_l_p.py.last_good_native
07:20:50  r_m_s_norm.py         + r_m_s_norm.py.last_good_native
07:22:21  rotary_embedding.py   + rotary_embedding.py.last_good_native
```

Six minutes, five components, all passing. **F4's fail-first/repair-on-retry design worked.** What
failed was reporting it.


---

## F5 — the systemic-pattern detector counts CLASS NAMES, so the broadest bugs are the ones it misses

**Status: FIXED, but LATENT — it was NOT the cause of this run's stall.** `termination_check` was
never callable (F6), so the systemic hint was never requested and this code never ran. The counting
bug is real and verified against our exact failure map, but it is insurance for future runs, not
the explanation for what went wrong here. The first write-up implied otherwise; corrected.

**Status: FIXED** (`bringup_mcp.py`) · severity: disables the tool's own escape hatch exactly when
it is most needed · reported: not yet

### The design being defeated

The agent prompt is explicit that per-component repair is the wrong response to a shared bug:

> If `termination_check` returns a non-null `systemic_hint`: STOP iterating per-component and
> address the shared root cause first. A systemic hint means 3+ components are failing with the
> same class — the fix belongs in `tests/pcc/conftest.py` or the common `_make_arg_for` helper,
> not in each stub. **Individual repairs will keep re-hitting the same wall.**

That is precisely our situation. It never fired.

### Why

```python
_hot = [(cls, cs) for cls, cs in _class_counts.items() if len(cs) >= 3]
```

It counts components sharing an **identical class string**. F4's single template produces two
different strings from the same line of code — `MISSING_KWARG` where a class needs more args,
`UNEXPECTED_KWARG` where a class rejects `mesh_device`:

```
MISSING_KWARG      attention, m_l_p              2
UNEXPECTED_KWARG   r_m_s_norm, rotary_embedding  2
SHAPE              decoder_layer                 1        -> nothing reaches 3, hint stays None
```

**The broader the shared bug, the more different symptoms it produces, and the less likely the
"same class" test is to fire.** A bug that breaks every class in the same way trips it; a bug that
breaks classes in *different* ways does not.

### Fix

Count by **family**, not by class name. `MISSING_KWARG`, `UNEXPECTED_KWARG` and `API_SIGNATURE`
are one family (`CONSTRUCTOR_SIGNATURE`) because they share a fix location. The hint also now
names the classes actually seen, and for this family points at the canonical-constructor call
rather than only at conftest.

### Verified — replaying the stalled run's exact failure map

```
OLD (by class name)   {'MISSING_KWARG': 2, 'UNEXPECTED_KWARG': 2, 'SHAPE': 1}  -> fires: False
NEW (by family)       {'CONSTRUCTOR_SIGNATURE': 4, 'SHAPE': 1}                 -> fires: True
```

---

## F6 — the agent's tool server never starts: `mcp` is an undeclared, missing dependency

**Status: FIXED (environment)** · severity: **the root cause of an 11-hour stall** · reported: not yet

### What the user sees

Nothing. That is the whole problem. Round after round of:

```
BRING-UP (cc) round 36 ... target=`?` rung=? (graduated 0) → invoke claude → gate
  · round 36 working… 45s, 8 tool calls
```

No error, no warning. Indistinguishable from a model that is genuinely hard to port.

### The actual cause

`_cli_helpers/bringup_cc.py` writes an MCP config telling Claude to launch the tool server:

```json
"command": "/opt/venv/bin/python",
"args": ["/…/scripts/tt_hw_planner/bringup_mcp.py"]
```

`bringup_mcp.py:65` does `from mcp.server.fastmcp import FastMCP` — an unguarded module-level
import. **`mcp` was not installed**, so the server died on startup and all ten tools silently did
not exist:

```
termination_check  list_components  run_component  record_result  restore_best
decompose_component  fall_back_to_cpu  mark_harness_skipped
resolve_reference_loader  get_shard_plan
```

`scripts/tt_hw_planner/` ships **no requirements file at all**. The only one in the PR
(`models/experimental/perf_automation/requirements-agent.txt`) lists `claude-agent-sdk`, not `mcp`.

### Every symptom this explains

| symptom | because |
|---|---|
| `target=?` `rung=?` | `termination_check` uncallable — nothing to name a target |
| `attempts` frozen at 1 | `record_result` uncallable — it is the only thing that bumps it |
| `graduated 0` while tests pass | graduation is recorded through `record_result` |
| 36 rounds × 45 s of no progress | the agent had only generic tools (see F4 correction) |
| the systemic hint never fired | it is returned BY `termination_check` |

### Fix

`uv pip install "mcp<2"` into the interpreter named in the config. **The pin matters**: the current
major renamed `mcp.server.fastmcp` → `mcp.server.mcpserver`, so a bare `pip install mcp` installs a
version that still fails the import. Installed 1.29.0; `from mcp.server.fastmcp import FastMCP`
then succeeds and `termination_check()` immediately returned `can_stop: True` with all five
components graduated.

### Suggested fix for the tool

1. Declare the dependency (`scripts/tt_hw_planner/requirements.txt`), pinned `mcp<2`.
2. **Pre-flight it.** `ttnn_preflight.py` already exists to check `import ttnn` before any device
   test; the same pattern applied to the MCP server would have turned an 11-hour silent stall into
   a one-line error at startup.

---

## F7 — all progress flows through one channel, and nothing notices when it is dead

**Status: OPEN (design)** · severity: converts any reporting fault into unbounded wasted time

F6 was survivable in principle: the agent finished the work anyway. What made it cost eleven hours
is that **the harness has exactly one way to learn anything** — the MCP tools — and no cross-check
against the filesystem it can see.

At 07:22 the work was complete and on disk: five repaired stubs, five `.py.last_good_native`
snapshots, tests passing. The harness's own graduation predicate is
`_is_graduated()` = *"snapshot exists AND stub is native"* — a pure filesystem check that would
have returned **True for all five**. It was simply never consulted outside the dead MCP path.

Meanwhile the loop's response to "no progress reported" is to start another identical round,
indefinitely (`max_consecutive_timeouts` defaults to 1000).

**Suggested:** re-derive graduation from disk at the top of each round, and halt with a loud error
after N rounds in which nothing on disk changed AND no tool call succeeded — "the agent reported
nothing and changed nothing" is a diagnosable state, not a reason to keep spending.

---

## F8 — a clean, deliberate refusal is misread as a hardware crash, and the card gets reset

**Status: OPEN (tool)** · severity: resets the user's device for a config error; burns 3 retries
· reported: not yet

### What the user sees

```
[optimize/cc] refusing to mutate an existing demo in place. Pass --in-place to override.
[optimize/supervisor] orchestrator exited rc=1 (likely native crash / device wedge)
                      -- resetting device + restarting (restart 2/3)
[optimize/supervisor] reclaimed device (killed holders none) + tt-smi -r 0 rc=0
```

The refusal is correct, intentional, and printed a clear message one line earlier. The supervisor
sees only `rc=1`, assumes *"likely native crash / device wedge"*, runs **`tt-smi -r 0` to reset the
accelerator**, and retries the identical command — three times, each ending the same way.

### Why it matters

Resetting hardware is not a neutral act: on this branch a board reset is the documented recovery
for a wedged card, and it interrupts anything else using it. Doing it in response to a *policy
refusal* is both useless (the refusal is deterministic — retrying cannot help) and disruptive.

`rc=1` here carries no information: the tool used the same exit code for "I decline" and "the
device died". The distinction exists one line above in plain text.

### Suggested fix

Give deliberate refusals a distinct exit code (or a sentinel line the supervisor greps for), and
only treat rc as a wedge when the device is actually unresponsive — which is cheap to test by
opening it. Verified after the three resets: the card was healthy the whole time
(`ttnn.open_device` fine, grid 13x10).

### CORRECTION (2026-08-13): the mechanism EXISTS — this is one wrong exit code

A later refusal in the same run printed:

```
[optimize/cc] refusing to start against a tool whose own tests fail.
[optimize/supervisor] child REFUSED to start (rc=3) — a decision, not a crash. Not restarting; the reason is above.
```

So the supervisor **does** distinguish a deliberate refusal from a crash, via a dedicated exit code
3, and correctly declines to reset the device or retry. The in-place refusal simply returns `rc=1`,
which falls into the crash path. **F8 is therefore a one-line fix — return 3, not 1 — not a design
gap.** The original write-up overstated it; corrected here.


---

## F9 — a local model directory is mistaken for a demo directory (same family as F1)

**Status: WORKED AROUND** · severity: `optimize` unusable when the model is a local path
· reported: not yet

`optimize` accepts either a model id or a demo directory, resolved by `_resolve_target`:

```python
p = Path(target)
...
if p.is_dir():
    return p.resolve()          # any directory is assumed to BE the demo
```

Our model is a local folder of weights, so passing it as the target made the tool treat
`/localdev/.../hf_models/voxtral-tts-backbone` as the demo to optimize. It is outside the repo, so
worktree isolation failed; it does not look planner-emitted, so `kind` became `"existing"`; and the
run refused (then F8 reset the card three times).

**Workaround:** pass the demo directory instead of the model id —
`optimize models/demos/voxtral_tts_backbone`. The classification then flips to `(emitted)` and it
runs in place, correctly.

**Suggested fix:** resolve a directory that contains `config.json` + weights as a MODEL (look it up
via `bringup_status.json` like the model-id path does), not as a demo. Same root cause as F1: local
paths are second-class throughout, and the failure surfaces far from the cause.

---

## S2 — OURS: `tt-perf-report` was never installed, so `optimize`'s preflight failed

**Status: FIXED (our environment)** · **NOT a tool defect — do not report**

### What we saw

```
[optimize/cc] preflight FAILED
  FAILED test_before_loop.py::test_before_loop_all_mocks_produces_manifest_and_baseline
  FAILED test_tracy_tool.py::test_tracy_tool_orchestrates_runs_and_median
  ... 4 failures
[optimize/cc] refusing to start against a tool whose own tests fail.
```

Root cause of all four: `FileNotFoundError: [Errno 2] No such file or directory: 'tt-perf-report'`.

### Why it is ours

`tt-perf-report==1.2.2` **is** listed in `models/experimental/perf_automation/requirements-agent.txt`,
with install instructions at the top of that file. We never ran it. Fixed with
`uv pip install -r models/experimental/perf_automation/requirements-agent.txt`; the tool's suite
then read **2617 passed, 7 skipped**.

### Contrast with F6 — this is the distinction that matters

| | F6 (`mcp`) | S2 (`tt-perf-report`) |
|---|---|---|
| declared anywhere? | **no** — `scripts/tt_hw_planner/` ships no requirements file | **yes**, with install instructions |
| detected? | **no** — silent, 11 h of empty rounds | **yes** — refused in seconds, named the tests and the override |
| owner | **tool** | **us** |

Same symptom class (a missing dependency), opposite verdicts. The difference is entirely whether
the tool declared it and checked for it.

---

## F10 — the F9 workaround loses the model id, so `optimize` cannot build its own correctness gate

**Status: WORKED AROUND** · severity: `optimize` unusable without a hand-supplied `--pcc-test`
· reported: not yet

### What the user sees

```
Step 6/10  Mapping the model's pipelines & building perf tests
  CANNOT CONTINUE — no usable correctness gate.
  no --pcc-test supplied, and no cached HF reference for None. There is no ground truth to check
  correctness against, so optimize would be free to commit edits that silently degrade the model.
  PLEASE GIVE A PCC TEST TO RUN OPTIMIZE: pass --pcc-test <file>::<test>.
```

**The refusal itself is correct and worth crediting** — it will not make a model faster if it
cannot prove the model is still right. The defect is the reason it got there.

### The actual cause — two findings interacting

Note `no cached HF reference for **None**`: the model id is `None`.

1. `optimize <model-id>` fails, because F9 resolves any directory argument as a demo dir and our
   model is a local folder.
2. The workaround is `optimize <demo-dir>` — which works, but **the model id is then never
   resolved**, so the stage that would auto-generate the PCC gate has no reference model to
   compare against.

So the documented "just point it at the directory" path cannot auto-generate a correctness gate,
and the model-id path that could is broken by F9. Either one alone is survivable; together they
close both routes.

### The gate existed the whole time

`emit-e2e` had already emitted `tests/e2e/test_e2e_pipeline.py`, which compares against the HF
golden, declares `PCC_THRESHOLD = 0.95`, and prints exactly the format asked for:

```python
print("e2e PCC=%s" % min(float(pcc_call1), float(pcc_call2)), flush=True)
```

The tool produced its own correctness gate one stage earlier and could not find it.

### Workaround

Pass it explicitly:

```
--pcc-test models/demos/voxtral_tts_backbone/tests/e2e/test_e2e_pipeline.py::test_e2e_pipeline
```

### Suggested fix

When the target is a planner-emitted demo, look for `tests/e2e/test_*.py` in that demo before
declaring there is no gate — the tool wrote it. And resolve the model id from
`bringup_status.json`, which sits in the same directory and records it.

---

## F8 (addendum) — a second refusal path also returns rc=1 and gets restarted

The `CANNOT CONTINUE — no usable correctness gate` refusal also exits **rc=1**, so the supervisor
retried it three times before giving up:

```
[optimize/cc] run failed (see messages above)
[optimize/supervisor] child exited rc=1; 3 restart(s) exhausted.
```

Same root cause as F8: deliberate refusals must return the dedicated refusal code (**rc=3**), which
the supervisor already handles correctly. At least two call sites return 1 instead — the in-place
refusal and this one.

---

## R — `optimize` RESULTS (running; snapshot 2026-08-13 20:04, 4h50m elapsed, ~2h30m in the loop)

Once the bring-up defects above were cleared, the optimize half ran unattended and **worked**. This
section is the credit half of the ledger and is the material for the comparison write-up.

### Headline

| | device_ms (whole test) | decode ms/token (trace+1cq) | e2e PCC |
|---|---|---|---|
| baseline as generated by `auto-up` | **1121.293** | 28.878 | 0.9976 |
| after the grid/structural rungs (20:04) | 348.026 | 23.168 | 0.9795 |
| after the dtype rung on MLP + attention | — | 18.282 | 0.9715 |
| after the down_proj shard-width sweep (O4b) | 281.8 | 16.763 | 0.9708 |
| after the same sweep generalised to K/V (O4b) | 276.7 | 16.135 | 0.9897 |
| after fusing Q/K/V into one decode projection (O4c) | 273.6 | 15.976 | 0.9708 |
| after fusing RoPE into one op (O4d) | 271.9 | 15.212 | 0.9903 |
| run 2 — Q+K rotated in one call (O4f) | 269.5 | 14.909 | 0.9903 |
| run 2 — decode-native Q/K/V layout (O4g) | 261.9 | 13.975 | 0.9903 |
| run 2 — head creation fed the projection's shard (O4h) | 254.5 | 13.277 | 0.9903 |
| run 2 — DRAM-sharded LM head weight (O4i) | — | 13.228 | 0.9904 |
| run 2 — untilize vocab blocks before joining (O4j) | — | 13.139 | 0.9904 |
| run 2 — fused K/V cache write (O4k) | 234.1 | 12.890 | 0.9904 |
| run 2 — decode SDPA 16 cores/head -> 2 (O4l) | 231.8 | 12.633 | 0.9904 |
| run 2 — fused-QKV grid re-swept 32 -> 48 (O4m) | 229.6 | 12.427 | 0.9903 |
| run 2 — SiLU folded into the SwiGLU multiply (O4n) | 229.0 | 12.352 | 0.9903 |
| run 2 — norm's shard chained into QKV (O4o) | — | 12.299 | 0.9903 |
| run 2 — gate/up plan, reshard now free (O4p) | — | 12.038 | 0.9903 |
| run 2 — residual stream kept in the norm's shard (O4q) | — | 11.928 | 0.9903 |
| run 2 — o_proj/down_proj hand shards to the residual add (O4q) | — | 11.839 | 0.9903 |
| run 2 — LM head vocab blocks written as bf8_b (O8) | — | **11.827** | **0.9774** |
| tool's own roofline target | 338.541 | — | gate 0.95 |
| **hand-port, for reference** | — | **15.907** | — |

**11.827 against 15.907 — the tool is 25.7% AHEAD of 74 human experiments** — but see O8: the last
step is where the run stops being worth watching., autonomously, with
PCC bit-identical across its last four wins (0.990347151783074, unchanged to every decimal). Every
one of those four was a layout or dispatch result. None spent accuracy.

State this with the accuracy bar attached, every time it is quoted: the tool is at e2e PCC 0.9708
against its 0.95 gate; the hand-port's p150 decode holds 0.981 (`STATUS.md`, and 0.99991 on the
N150 branch). Those two numbers are not measured over the same thing, so they are not a clean
comparison — but the tool is certainly not *tighter*, and part of the last 3 ms came from
`bfloat8_b` on `down_proj`, which is §6.16's `w2` decision the hand-port took and then handed back.

**On the dtype walk, and a correction.** Three commits took `gate/up`, `down_proj` and `q/k/v/o` to
`bfloat8_b`, and PCC did walk 0.9795 → 0.9715 alongside 23.168 → 18.282. But the ladder then
stopped itself: pushing `q_proj` on to `bfloat4_b` measured **faster** (18.282 → 17.770) and was
**reverted on PCC 0.7707**, with reasoning worth quoting —

> *q_proj is the most exposed weight in the block for this lever: its output goes through RoPE into
> the attention scores, so a coarser weight perturbs WHICH positions attend to which, not just by
> how much — a change the softmax then amplifies.*

That is the same argument §6.17 makes about top-2 gaps in a discrete decision, reached
independently. PCC has since recovered to 0.9897. So the fair statement is **not** "it trades
accuracy until the gate stops it" — it stopped one rung early, on structure, and its best result is
its most accurate recent one. O1 still stands as a gap (there is no per-weight axis, and
`down_proj → bf8_b` is the §6.16 `w2` decision the hand-port took and then deliberately gave back),
but the ladder is more discriminating than the raw PCC trend suggested.

**3.22× on device time, autonomously, with the PCC gate held the whole way** (0.9795 against its
0.95 e2e threshold; one attempt fell to 0.8638 and was correctly rejected and reverted).
`modeled_floor_ms` 400.68; throughput 43.16 tok/s against a 71.31 theoretical, now `IN_BAND`.

### Against the hand-port (this is the number the experiment was for)

| Block 1 decode step | ms | source |
|---|---|---|
| N150 branch this port forked from | 23.15 | `STATUS.md` header table |
| **tool, fully autonomous, ~2.5 h in the loop** | **23.168** | `perf_mcp_stage_ms`, trace+1cq |
| hand-port at §6.39 (p150 fork, pre-§6.65/§6.67) | 21.2 | `STATUS.md` header table |
| **hand-port current, §6.72** | **15.907** eager / **15.922** traced | `STATUS.md:4148`, `:4699` |

**The tool is at the hand-port's starting line, not its finish line.** 23.168 vs 23.15 is a dead
heat with the N150 build the p150 port forked from — i.e. it independently reached in one
afternoon what that branch already had — but the hand-port's *current* Block 1 is **15.9 ms**, so
the tool is **1.46× slower** than the target stated at the top of this file. It has not closed the
gap; it has covered the first third of it.

Two things separate them, and only one is a fair fight:

1. **Precision.** The tool is buying part of its 23.168 with trades the hand-port refused —
   `bfloat4_b` on the LM head and a LoFi compute config — against a hand-port whose decode holds
   PCC 0.981–0.99991. Its 0.95 e2e gate permits what §6.16/§6.17 rejected on quality grounds. So
   the honest read is that the tool is 1.46× slower *and* less accurate, not trading one for the
   other.
2. **The rungs it has not reached.** It is still on `knob:grid` and `knob:dtype` for the remaining
   matmuls. The hand-port's last 5 ms came from §6.65/§6.67-class structural work plus per-weight
   precision (§6.16), and the per-weight axis does not exist in this tool (O1).

*(Corrected 2026-08-13: an earlier draft of this section compared against the stale 21.2 ms figure
from the `STATUS.md` header table and claimed the tool was within 9.3%. §6.72 superseded that
number; the header table was not updated. The real gap is 46%.)*

### What it found, in order (14 commits)

```
argmax        untilize logits to ROW_MAJOR              1121.3 -> 545.7   (-51.3%)
lm_head       bf8_b weight, then bf4_b, then LoFi        545.7 -> 397.3
rmsnorm       width-shard the decode norm                397.3 -> 370.4   (-2.11 ms/token)
matmul        full-grid 1D-mcast plans, q/k/v/o/down     370.4 -> 356.7
host          capture the decode step, replay per token  356.7 -> 354.9
datamove      producers emit the consumer's shard        354.9 -> 348.0
```

### Independent rediscovery of the hand-port's findings

Three of the hand-port's recorded wins were re-derived from scratch by a tool that had never seen
that code:

- **`perf(rmsnorm): width-shard the decode norm so it fills the grid`** — the hand-port's **§6.67**,
  its single largest win at −5.399 ms/frame. Same lever, same stated mechanism (the norm was at
  `grid=tiny`, one core).
- **`perf(decode): capture the decode step once and replay it per token`** — **§6.65**, −4.244 ms/frame.
- **the five full-grid decode projection plans** — **§6.52**, −4.24 ms/frame.

Note §6.67 was a *reversal* — the hand-port shipped the sharded norm, reverted it at §6.39, then
reinstated it at §6.67. The tool went straight to the end state.

### O4 — the argmax finding is new, and it explains a hand-port rejection

The first and largest single win, worth **51.3% of device time**, is a cause the hand-port measured
but never diagnosed. The tool's note:

> `ttnn.argmax` picks its parallel path from INPUT LAYOUT, not a flag — `uses_multicore_path()`
> (`argmax_device_operation.cpp:16`) bails to the single-core kernel for any non-ROW_MAJOR input,
> and `ttnn.linear` hands it TILE, which also pads the `[1,1,131072]` decode row to the 32-row tile
> height so one core scanned ~32× the needed bytes.

`STATUS.md` §6.8 measured exactly this pathology on Block 2's `semantic_code` — *"argmax over 8320
values, 490.1 us, 39.9%, 33 KB at 0.07 GB/s — ALL overhead"* — and worked around it by moving the
reduce to the host (option C, 1.439×, shipped). **That rejection is worth re-running with a
`ttnn.to_layout(logits, ROW_MAJOR)` in front of the device argmax.** The host path may still win on
a 33 KB reduce that already ends in a D→H copy, but the A/B was scored against a single-core kernel
that did not have to be single-core.

This is the clearest case in the whole experiment of the tool contributing something the human pass
missed, and it came from reading the kernel's C++ dispatch condition — not from a sweep.

### O4b — "fill the grid" is WRONG for a K-heavy decode projection, and this is the second finding to take

The second-largest win of the run, and it is precision-neutral — a pure shape result. Having earlier
committed full-grid plans for every decode projection, the tool contradicted its own heuristic on
`down_proj` (`32 x 9216 x 3072`, the deepest K-reduction in the model). Its diagnostic: the op takes
**the same 0.159 ms/call at bf16 and at bf8_b**, so it is not bandwidth-bound and the limit is the
shape of its k-reduction, not the bytes. It then swept the activation shard width — which sets
`in0_block_w` and `per_core_N` together — instead of assuming the widest grid:

```
cores    96      48      32      24      16      12       8
ms    0.1589  0.0960  0.1137  0.0934  0.1084  0.1342  0.1964
                               ^ pinned
```

Non-monotonic, with a **1.70× spread** between the widest grid and the best one. Kept:
`18.282 → 16.763 ms/token` (−8.3%), PCC held at 0.9708. Its own note calls it *"a correction to my
own earlier heuristic: 'occupy the full grid' is wrong for a K-heavy decode projection."*

**Why this matters for the hand-port.** `tt/ttnn_voxtral_gpt.py:76` defines **one** grid for all
five decode matmuls:

```python
_MM_GRID = (12, 6)      # 72 of the 130 cores; 13x10 measured 0.31 ms WORSE
...
_PRG_W2 = _mm1d(4, 2)   # K=9216  N=3072  Nt=96 -- the deepest reduction in the model
```

§6.52 swept that choice **globally** (72 against 130, −4.24 ms/frame) but never **per-op**. `w2` is
the identical op to the one swept above — same K, same N — and it is running on 72 cores, which
sits on the wrong side of the curve the tool measured (48c 0.0960, 96c 0.1589). The other four
projections are K-light and may well want the wide grid they have; the point is that one global
`_MM_GRID` cannot be right for both.

**Concrete suggestion: give `_PRG_W2` its own `compute_with_storage_grid_size` and sweep it
independently.** The tool's curve says the win is in the 24–48 core range and is worth ~1.7× on
that op.

**It then generalised the lesson itself**, re-sweeping K/V at the current dtype rather than
trusting the widest even split:

```
K/V (32 x 3072 x 1024)   default   32c      16c      8c       4c
                          0.0528  0.0379  0.0251  0.0314  0.0535 ms
                                          ^ pinned                  a further -34%
```

`16.763 → 16.135 ms/token`, and **PCC improved** 0.9708 → 0.9897. Q measured best at 32 cores, so it
no longer shares a grid with K/V at all.

**Caveat on how far this transfers — worth stating before acting on it.** The tool's model has
`q_proj` and `k/v` as separate ops; the hand-port **fuses** them into one `wqkv`
(`_PRG_QKV = _mm1d(2, 3)`, K=3072 N=6144). A per-projection grid cannot be applied to a fused QKV
without unfusing it, and unfusing costs a dispatch the hand-port deliberately paid once. So:

- **`w2` / `down_proj` — transfers directly.** Standalone op in both, identical shape. Act on it.
- **Q/K/V — does not transfer as-is.** It is evidence that the *optimum differs per projection*
  (Q wants 32, K/V want 16), which is an argument about whether the fusion is still worth it, not a
  drop-in change.

**SUPERSEDED within the hour, by the tool itself.** The very next attempt was the structural rung on
QKV, and it independently arrived at the hand-port's design: stage the three weights **concatenated**
into one 3072×6144 tensor and run **one wide matmul per token** instead of three, opening one shard
instead of two, then slicing. `16.135 → 15.976 ms/token`. Its own explanation of why the win is small
is the interesting part:

> *Modest rather than dramatic because the three separate projections were already individually tuned
> (their own core counts and a shared input shard), so what is left to win is the per-op dispatch and
> one reshard, not bytes — the fused read moves exactly the same weight bytes.*

So the caveat above was right about the mechanism and wrong about the conclusion: the per-projection
optima are real, and fusing still wins anyway, because what fusion buys is dispatch, not bandwidth.
The hand-port's fused `wqkv` is vindicated by a tool that tried it both ways.

It also recorded what it deliberately did **not** take: `nlp_create_qkv_heads_decode`, which would
collapse the 3 reshapes + 3 permutes as well, because it emits `[1,B,H,D]` while this SDPA path
consumes `[1,H,S,D]` — a contract change across the cache write and SDPA, so it belongs in its own
attempt rather than riding along. That is the same boundary §6.68/§6.72 negotiated by hand.

### O4d — RoPE as one op, and where the tool's remaining lead actually comes from

The generated stub wrote `x*cos + rotate_half(x)*sin` out longhand — two slices, a neg and a concat
to build `rotate_half`, then two multiplies and an add. **Seven dispatches for one elementwise
rotation, twice per layer, 26 layers ≈ 360 launches per token** against tensors of a few KB. That is
why the roofline tagged this model's `BinaryNg`/`Reshape`/`Concat`/`Slice` ops `bound_by=dispatch`.
It replaced the chain with `ttnn.experimental.rotary_embedding_hf`: `15.976 → 15.212 ms/token`
(−4.79%), **and PCC rose** 0.9714 → 0.9903, because the fused kernel accumulates the rotation
internally instead of round-tripping two bf16 products through DRAM.

Two things in that commit are worth lifting on their own:

- **It explained a metric disagreement rather than picking the flattering number.** *"device_ms
  barely moves because it SUMS per-op durations — what shrank is the inter-op gap those ops were
  tagged for, which only the wall metric sees."* 273.62 → 271.87 device, against −4.79% wall.
- **It shipped a latched fallback** — the explicit chain is retained for operands the fused op
  refuses, and the flag latches after the first refusal *"so it cannot raise inside a captured
  trace."*

**This one is NOT a finding for the hand-port — it is the tool catching up.** `ttnn_voxtral_gpt.py`
has used `rotary_embedding_hf` from the start (`_rope`, and both decode call sites). Worth noting
the hand-port is still ahead on this specific op: it calls the **decode-specialised** path
(`is_decode_mode=True`, with cos/sin sharded to match, per the comment at line 53), where the tool
calls prefill mode with `s=1`.

#### So where does the tool's 0.7 ms lead actually come from?

Now that both implementations agree on fused QKV and fused RoPE, the remaining delta is a short
list — and it splits cleanly into "free" and "paid for":

| the tool has | hand-port equivalent | free? |
|---|---|---|
| per-op narrow grids: `w2` 24c, K/V 16c | one global `_MM_GRID` (12,6) = 72c for all five | **FREE — precision-neutral. This is the one to take.** |
| `down_proj` at `bfloat8_b` | `w2` at bf16 **on purpose** (§6.16) | PAID — 77% of the precision stack's accuracy cost |
| LM head at `bfloat4_b` | n/a — Block 2 consumes the hidden state | n/a |
| `ttnn.argmax` on a ROW_MAJOR input | argmax on the **host** (§6.8) | FREE — but re-measure, see O4 |

**The conclusion for the comparison write-up:** roughly the whole of the tool's lead is (a) per-op
grid sweeps the hand-port never ran, which cost nothing in accuracy, and (b) one precision trade the
hand-port evaluated and deliberately declined. Take (a); (b) is already settled and settled
correctly.

### F11 — the documented `--max-rounds` default is 20; the real one is 3, and it is the ONLY exit

**Status: found 2026-08-13, run 1 ended on it** · severity: silently caps every run at ~1/7 of the
advertised effort · reported: not yet

Run 1 ended after 7h20m with this line, and nothing else:

```
pipeline main: 3 round(s), can_stop=False
```

`can_stop=False` is the tool's own gate saying **it should not have stopped**. It stopped because
`DEFAULT_MAX_ROUNDS = 3` (`cc_optimize/run.py:39`, and `cli.py`'s `--max-rounds` default). The
documentation says otherwise:

```
GETTING_STARTED.md:272
  `--max-rounds N` — cc engine: max `claude -p` optimization rounds per pipeline (default `20`).
```

**20 documented, 3 in code.** A user who reads the guide and does not pass the flag gets 3 rounds.

**This compounds with O7 and that is the real severity.** O7 records that the throughput band can
never fire `can_stop` when the parameter count is estimated — which is every locally-supplied
model. So for a local model the band exit is unreachable, the floor exit is unreachable, and
`--max-rounds` is the *only* thing that ends a run. The documented behaviour ("the deterministic
gate can still stop earlier once each op is at its floor") is therefore not what happens: **every
such run ends at exactly 3 rounds, mid-climb, and reports it in one line at the bottom of a long
report.** Ours had just produced its two largest wins in round 3 when the cap hit.

**Fix:** align the default with the documentation (or the documentation with the default), and make
the terminal line say *why* it stopped — `stopped: round cap (3) reached with can_stop=False` reads
very differently from `3 round(s), can_stop=False`.

**Verified:** relaunched with `--max-rounds 20`, pid 1067756, from the same tree.

### O4e — the tool independently corroborates §6.72, the hand-port's most contested experiment

In its last round it tried `nlp_create_qkv_heads` with `transpose_k_heads=False` to replace the
head-split's 9 ops (3 slices + 3 reshapes + 3 permutes) with one fused call. PCC was bit-identical,
so this was a pure data-movement question — and it **regressed**: device 273.62 → 281.07 (+2.7%),
per-token 15.21 → 16.19 (+6.45%). Its conclusion:

> *Dispatches fell 3413 → 2867 yet it got slower — decisive: these were view ops doing no work.*

That is §6.72 reached from the other direction. The hand-port went fused → hand-rolled and measured
−0.775 ms/frame bit-exact; the tool went hand-rolled → fused and measured +0.98 ms/token. Both land
on the same conclusion, and the tool's phrasing supplies the mechanism §6.68 got wrong when it
"counted one op short": **dispatch count is not the cost when the ops being removed are views.**

Two independent confirmations of a reversal that was the hardest call in the hand-port's log.

### O4f — a THIRD finding to take: the same win §6.23 rejected, by a route that avoids what it rejected

Run 2's first win, and the one most worth acting on, because it looks like a contradiction of the
hand-port and is not one.

**What the hand-port rejected (`NOTES.md [gpt-23]`, §6.23):**

> *Two calls, not ttnn's fused q+k rope: that one implements the INTERLEAVED convention via a
> trans_mat, and our wq/wk are permuted to HALF-SPLIT at load. Measured 0.236 ms/frame for reverting
> that permute, disjoint q/k cores and losing bit-exactness.*

A correct rejection. The objection is to **ttnn's fused q+k rope operator**, whose convention
disagrees with the half-split layout the weights are permuted into at load.

**What the tool did instead:** it did not use that operator at all. Q and K are adjacent in the
fused QKV output and RoPE applies per head against the same cos/sin, so it slices **q+k out
together**, splits heads once into a **40-head** tensor (32 q + 8 kv), calls the *same*
`rotary_embedding_hf` **once**, and slices query/key apart afterwards.

```
trace+1cq 15.2117 -> 14.9090 ms/token (-1.99%)    device 271.92 -> 269.45
PCC 0.990347 -> 0.990347, UNCHANGED               dispatches 3413 -> 3257
```

Same convention, same op, one dispatch instead of two — plus it drops a slice and a reshape+permute
pair, so the block is 2 ops lighter per layer on top of halving the rotary count. **The fusion is
exact, not an approximation**, which the unchanged PCC confirms.

**Why this matters:** §6.23 measured the cost of a *convention change* and rejected it, correctly.
This route asks nothing of the convention. The hand-port keeps half-split, keeps its permute, keeps
bit-exactness, and still gets the launch-count win — 52 rotary launches per token become 26. For
scale, `[gpt-24]` measured a comparable 26-launch saving (the fused KV write) at **0.405 ms/frame,
bit-identical**.

**The one thing to check before taking it.** The hand-port's decode RoPE runs `is_decode_mode=True`,
and `ttnn_voxtral_gpt.py:53` records that *"rotary_embedding_hf's decode mode requires cos/sin
sharded as well as the input"*. Concatenating to 40 heads changes the shard shape, so the question
is whether a 40-head sharded rotation is expressible on this grid — not whether the arithmetic
holds, which it does. (The tool calls prefill mode with `s=1`, so it never met this constraint.)

**Status: strongest candidate of the three.** Unlike O4 (argmax, needs a re-measurement) and O4b
(`w2` grid, needs a sweep), this one is a known quantity: exact arithmetic, a measured win on the
identical model, and it sidesteps the specific objection that got it rejected the first time.

### O4g — decode was wearing prefill's layout, and 31 of every 32 rows were padding

The largest run-2 win, and the one with the most general lesson in it. The block shaped Q/K/V as
`[1, heads, seq, hd]` because that is what a **prefill** SDPA reads — but nothing in a decode step
reads that. `paged_update_cache` wants `[1, batch, kv_heads, hd]`; `scaled_dot_product_attention_decode`
wants `[1, batch, q_heads, hd]`. Bridging the two cost a reshape, a permute and three slices up
front, a permute plus a reshard before each cache write, and a permute in and out of the attention:

> *19 ops per layer where 8 do the work, and the profiler tags every one of the other 11
> `bound_by=dispatch`: their cost IS the launch.*

`nlp_create_qkv_heads_decode` emits that layout directly out of the fused projection, L1-sharded, so
K and V arrive already in the memory config the cache write takes and Q already in the one the
decode SDPA takes — no permute at either end. **`14.909 → 13.975 ms/token` (−6.3%), device 269.45 →
261.85, PCC bit-identical.**

Note this is the op it *declined* two attempts earlier as "a contract change across the cache write
and SDPA, so it belongs in its own attempt rather than riding along." It came back for it. The
deferral was not avoidance.

**The general lesson — and the tool has now found it twice.** Buried in that commit:

> *That also drops the rotation from 160 tiles to 4: in the old layout 31 of every 32 rows were
> tile padding.*

A decode step is one row. In TILE layout one row occupies a 32-row tile, so **any decode-path op
that has not been given a decode-shaped input does 32× the work it needs to.** This is the same
finding as O4 (`ttnn.argmax` scanning ~32× the bytes because `ttnn.linear` handed it a TILE row).
Two independent discoveries of one pathology in one run.

**Worth a systematic pass on the hand-port:** for every op on the decode path, check whether its
input is a padded tile row. It is a class of waste that profiles as "this op is slow" rather than
"this op is doing nothing," which is why it survives casual inspection.

#### An apparent fourth agreement — WITHDRAWN two commits later, by the tool itself

It initially declined `paged_fused_update_cache`, reasoning that the op parallelises K and V across
**disjoint cores** and rejects operands sharing one, and at batch 1 head-creation puts K and V on
the same single core. That matched `NOTES.md [gpt-19]` / §6.44, which **deleted** `_V_SHARD` —
existing solely to let that op accept K and V — because on Blackhole *"that fused write is 0.687
ms/step SLOWER than two plain writes."* I recorded it as a fourth independent agreement.

**It is not. See O4k — the tool came back and landed the fused write.**

**This also arms O4f's open question.** §6.44 records the silent failure mode that went with
`_V_SHARD`: *"RoPE on a core whose cos/sin table lives elsewhere returns 3.4e38 from uninitialised
L1."* That is precisely the hazard waiting for anyone sharding a 40-head rotation per O4f — the
hand-port's own notes already document the trap.

### O4h — the bottleneck the previous fix created, and the loop catching it

Immediately after O4g, head creation became the largest open gap — an op producing **12 tiles** was
costing **33 µs**. The diagnosis:

> *`nlp_create_qkv_heads_decode` is batch-parallel, so at batch 1 its outputs live on ONE core — and
> handed an interleaved operand, that one core had to pull the whole fused row, 192 tiles, out of
> DRAM by itself.*

The projection had just written that row across 32 cores' L1, and the op's sharded program factory
reads a width-sharded operand from exactly there. Feeding it directly turns a single-core DRAM read
into a fan-in over L1:

```
head creation   33 -> 5.9 us/call   (8.44 -> 1.50 ms)
rotation         7.35 -> 3.14 ms    (follows onto the same shard)
datamove        26.8 -> 14.6 ms
                13.975 -> 13.277 ms/token (-5.0%)   PCC bit-identical
```

This is the measure → attack → re-measure loop doing exactly what it is supposed to: the previous
win moved the bottleneck, and the next round found where it went.

---

## ★ THE PATTERN — one pathology, found three times, and the best thing to take from this experiment

Three of the tool's largest wins are the same underlying bug wearing different clothes. **Ops
written for a batch or sequence dimension degenerate when decode gives them neither.** Each profiles
as "this op is slow", never as "this op is doing nothing", which is why all three survived a human
pass:

| where | what decode gave it | what it did | cost |
|---|---|---|---|
| **O4** `ttnn.argmax` | a TILE row | took the **single-core** path (dispatch keys off input LAYOUT) and read the row padded to 32 | 51.3% of device time |
| **O4g** RoPE / rotation | a TILE row | processed **160 tiles where 4 had data** — 31 of every 32 rows were padding | part of −6.3% |
| **O4h** `nlp_create_qkv_heads_decode` | batch 1 | **batch-parallel** op collapsed to ONE core, which then pulled 192 tiles from DRAM alone | 33 µs for a 12-tile op |
| **O4j** LM-head vocab-block join | a TILE row | concatenated 4 blocks of **2 MB where 64 KB was real** — again 31 of 32 rows | 0.080 ms/token, over half of O4i's win |

The unifying rule: **at batch 1, seq 1, a "parallel" op may be running on one core, and a "small"
tensor may be 32× its logical size.** Neither is visible in the source — both are properties of what
the op does with the shape it is handed.

**O4j is the clearest illustration**, because the fix costs nothing at all. Only row 0 of those
blocks is real — the decode activation is one token padded to the tile — so untilizing each block
*before* the join turns it from 2 MB into 64 KB and the join moves **256 KB instead of 8 MB**. And
it is not even an added step: `_greedy_token` already has to untilize to reach the multi-core argmax
path (O4), so this only moves existing work in front of the join instead of behind it. **The O4 fix
is what made O4j free** — the wins compound.

**Recommended action on the hand-port:** a systematic decode-path audit against this rule. For every
op in the decode step, ask (a) does its parallel path key off layout or batch, and does decode
satisfy that? and (b) is its input a padded tile row? The tool found three instances in one model;
there is no reason to think a hand-written port has zero.

### O4i — a real bandwidth finding, on a trade the hand-port should NOT take

Diminishing returns begin here: **−0.4%** (13.277 → 13.228). The diagnosis is still good —

> *The LM head was the worst matmul in the model: 226 MB of bfloat4_b weight at **256 GB/s**, half
> the board's DRAM bandwidth, where the layer projections manage **340–360**. The reason is
> placement, not size — an interleaved buffer round-robins its pages across all eight banks, so each
> of the 128 cores gathers its column slice from every bank at once.*

Width-sharding the weight in DRAM makes each core's slice contiguous in **one** bank, which is what
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` reads. Genuinely useful knowledge: an
interleaved DRAM weight can cost ~30% of achievable bandwidth purely through bank placement.

**But the cost is a second copy of the weight — 226 MB — because the DRAM-sharded kernel requires an
in0 height of exactly one tile, so prefill (logits at every prompt position) must keep the
interleaved copy.** The tool judges that against 32 GB of board DRAM and takes it.

**Do not transfer this one.** The hand-port runs three blocks co-resident on the same board and
treats headroom as a real budget; spending 226 MB for 0.05 ms/token is the wrong side of that trade,
and Block 1's LM head is not even on the hand-port's critical path (Block 2 consumes the hidden
state, per §6.8's semantic head). Recorded as a **bandwidth lesson worth keeping and a change worth
declining** — the placement insight generalises to any large interleaved weight; the duplication
does not.

### O4k — §6.44's reversal may itself be CONDITIONAL, and this is the finding with the most at stake

**`12.890 ms/token` (−1.9%), PCC unchanged.** The tool went back to the fused cache write it had
declined, and its solution is verbatim the hand-port's own N150-era trick:

> *Moving V one core over is enough to make it legal, and a 2 KB shard move is far cheaper than the
> launch it buys back.*

Compare `NOTES.md [gpt-24]`: *"V is moved to core (1,0) first because the op refuses an overlap."*
Same fix, found independently.

**The conflict.** §6.44 measured the fused write **0.687 ms/step SLOWER** on Blackhole and deleted
the machinery. The tool measures it **faster** on Blackhole — 0.402 → 0.210 ms/token across 26
layers. Both are Blackhole p150b. Both are this model.

**CORRECTION (2026-08-14).** I first proposed that §6.44 was conditional on a layout the hand-port
had not adopted — the decode-native `nlp_create_qkv_heads_decode` output that O4g introduced — and
called this the highest-value item to test. **That mechanism is wrong.** Reading
`ttnn_voxtral_gpt.py::_layer_step` directly: the hand-port **already calls
`nlp_create_qkv_heads_decode`**, and already feeds it a sharded operand
(`to_memory_config(reshape(qkv), _QKV_SHARD)`). It has both halves of what I claimed it lacked.

So the honest position is narrower and more interesting: **two careful measurements of the same
change, on the same hardware, against materially similar layouts, disagree.** §6.44 records the
fused write losing 0.687 ms/step; the tool records it winning 0.402 → 0.210 ms/token. Both moved V
one core to satisfy the disjoint-operand rule. I cannot account for the difference from the
artifacts I have.

Still worth an A/B — a 0.19 ms/token swing is real either way — but as an *unexplained
disagreement*, not as a mechanism I have identified. Downgraded accordingly in the ranked list.

#### A fifth agreement, in the same commit's rejected attempt

It tried fusing the MLP's SiLU into the gate projection and reverted it (+13 ms):

> *`activation=` alone does not fuse — with no program config ttnn appends a `unary_chain` op, which
> is the same launch under another name.*

`NOTES.md [gpt-26]` says exactly this: *"`activation="silu"` never fused; `fused_activation` does."*
Independently confirmed. It also added a detail the hand-port's note does not have: naming a core
grid to reach the fused path made gate/up **43.35 → 57.50 ms on the same 96 cores**, because *"the
router's auto-derived config for a named grid is not the one it picks for itself."*

### O4l — a SIXTH agreement, arrived at through a different knob

Handed no program config, `scaled_dot_product_attention_decode` spends the whole grid: at batch 1
with 8 KV heads that is **16 cores per head, 128 active**. Its diagnosis is the pattern again —

> *Sixteen ways is far too fine for a 256-deep cache — 16 positions per core, half a tile of work —
> and the kernel then pays a 4-round tree reduction ACROSS those cores to put each head back
> together. **The reduction, not the read, was what the op cost.***

It swept `max_cores_per_head_batch`: `default(16) 5.29 / 4 → 3.25 / 2 → 3.04 / 1 → 2.99 ms`.
Monotone, *"which is the tell that the reduction dominates throughout."* `12.890 → 12.633 ms/token`.

**The hand-port is already there, by another route.** `_SDPA_PRG` passes
`compute_with_storage_grid_size=CoreCoord(8, 2)` to the decode SDPA — 16 cores total, 8 KV heads,
so **2 cores per head**. The tool swept a per-head knob and landed on **2**. Same effective
parallelism, reached from opposite directions. (`[gpt-21]` records the hand-port also found faster
settings that were *"NOT SAFE — position sweep"*, a correctness dimension the tool's single-length
gate cannot see.)

Worth noting the tool took **2 rather than 1** despite 1 measuring faster: *"the last step is worth
0.05 ms, whole-model came out marginally better at 2, and 2 still has somewhere to go when a deeper
cache makes the read matter again."* It declined 0.05 ms to keep headroom at longer context — a
judgment about future conditions, not a greedy pick.

### O4m — a structural change silently invalidated an earlier knob, and it went back for it

> *The 32-core cap was tuned when Q, K and V were three separate projections. Fusing them into one
> 3072→6144 read widened N and moved the optimum, but the cap stayed where it was.*

Re-swept: `16 → 17.47 / 32 → 15.98 / 48 → 13.61 / 96 → 35.71 ms`. **48 wins by 15% over the value in
place.** And the cliff at 96 is the pathology `down_proj` already documented in O4b: `in0_block_w`
falls to 1, so each core walks 96 sequential single-tile k-blocks to produce 2 output tiles with
nothing to overlap the reduction against.

**The meta-lesson, and it applies directly to the hand-port: a structural change invalidates every
knob tuned before it, silently.** Nothing errors; the old value simply stops being optimal. The
hand-port's `_MM_GRID = (12, 6)` was fixed at §6.52 and has since been through §6.65 (traced loop),
§6.67 (sharded norm) and §6.72 (head split). It has not been re-swept since. That is the same
staleness this commit found, and it sharpens O4b: **re-sweep against the current structure, not from
the historical setting.**

### O4n — the hand-port's answer is better, and the tool's own note says why it couldn't reach it

The SiLU was a standalone unary fetching 9216 values out of DRAM and writing them back *"to do
almost nothing — 26 launches per token."* The tool's fix: the product is its only consumer and a
binary op can apply a unary to an input **as it reads it**, so `silu(gate) * up` becomes one launch.
`12.427 → 12.352`.

**The hand-port does not have that unary at all.** `_PRG_W1 = _mm1d(2, 4, UnaryWithParam(SILU))`
puts `fused_activation` inside the matmul program config, so the activation happens **in the
projection**, with no separate launch to fold anywhere. That is strictly better than folding it into
the consumer.

**And the tool explains its own gap**, in the attempt it reverted:

> *Deliberately not `ttnn.linear(activation="silu")`: with no program config to put it in, ttnn
> appends a `unary_chain` op — the same launch under another name — and **naming a core grid** to
> reach the genuinely fused path costs far more than the unary did (measured: gate/up 43.35 → 57.50
> ms on the same 96 cores, because the router's auto-derived config for a named grid is not the one
> it picks for itself).*

That is the whole story: the tool reaches for fusion by **naming a grid and letting the router
derive a config**, and the derived config is worse than the router's own default. The hand-port
writes the **full** `MatmulMultiCoreReuseMultiCast1DProgramConfig` by hand — `in0_block_w`,
`per_core_N`, `out_subblock_w`, and `fused_activation` together — so it gets the fusion without
inheriting a bad config. `NOTES.md [gpt-26]` records exactly this: *"`activation="silu"` never
fused; `fused_activation` does."*

**→ Tool defect, and a concrete one for the PR (see F12).** Not a wrong answer — a missing lever.

### F12 — the fusion rung reaches for a grid when it should reach for a program config

**Status: found 2026-08-14** · severity: leaves activation fusion unreachable, and misprices it as a
loss · reported: not yet

When the ladder wants to fuse an activation into a matmul, it does so by naming a core grid and
letting ttnn's router derive the rest of the program config. Measured cost on this model: **gate/up
43.35 → 57.50 ms on the same 96 cores** — a 33% regression that has nothing to do with the fusion
and everything to do with the derived config. The attempt is then correctly reverted, and the
catalogue records activation fusion as a **loss**, when the lever simply was not pulled.

**Fix:** when fusing an activation, emit a complete program config (`in0_block_w`, `per_core_M/N`,
`out_subblock_h/w`, `fused_activation`) rather than a grid. The tool already builds exactly such
configs on the `grid` rung — O4b/O4m sweep `in0_block_w` and `per_core_N` directly — so the
machinery exists; the fusion rung just does not use it. The hand-port reached +0 launches this way
and the tool reached +1; the difference is one code path.

### O4o — a local sacrifice for a global win, plus a ttnn fact worth knowing

> *At decode the sharded norm converted its result back to interleaved on the way out and the
> projection converted it straight back in — two launches per layer for a tensor that never needed
> to leave L1.*

The fix requires both to agree on a grid, so the norm **moves off its own optimum** — swept
`8 → 4.15 / 32 → 3.72 / 96 → 4.35 ms`, an interior minimum at 32 — and pays `3.72 → 3.98 ms` at 48
to sit on the projection's grid, against **0.80 ms of layout ops removed**. Taking a 0.26 ms local
loss for a 0.80 ms global win is a trade a per-op ladder is not obviously able to make, and it made
it.

**The ttnn fact, which is load-bearing and not obvious:** `to_memory_config` does **not** treat
"already in the requested config" as a no-op — **it dispatches a copy**. Any code that defensively
normalises memory configs is paying for launches it may not need.

**This is take-item #4 for the hand-port** — `_norm` returns `sharded_norm(x, gamma, NORM_EPS, _L1)`,
i.e. it shards internally and hands back **L1 interleaved**, and `_layer_step` feeds that straight
into `ttnn.linear(..., DECODE_PRG["wqkv"])`. Same round trip. See the ranked list for the grid catch.

#### And [gpt-28] is the precedent that makes re-opening rejections reasonable

The hand-port has already lived through exactly this. `NOTES.md [gpt-28]`:

> *the decode RMSNorm is width-sharded again, +5.399 ms/frame. **6.39/6.40 rejected this at +4.4 ms
> WORSE, but that cost was the RESHARD DISPATCH, which 6.65 traced away.***

A rejection that was correct when measured, invalidated by a later, unrelated change, and reversed.
That is documented precedent in the hand-port's own log for the general claim behind re-open items
#4–#6: **a rejection is only valid under the conditions it was measured in, and structural changes
move those conditions.** §6.67 is the proof that this repo already knows it.

### O4p — and the thing this whole run is actually demonstrating

> *gate/up kept ttnn's default routing on the strength of an old sweep: seven core counts, all lost,
> "the plan's two reshard ops cost more than they buy". **That accounting was correct and is now
> obsolete.*** *The norm ahead of this block emits its result IN a 48-core width shard, and gate and
> up read the SAME activation — so the plan's input reshard is not two ops, or even one. It is zero,
> and what is left is the matmul routing on its own, which is the part the old sweep could never see
> separately.*

`43.28 → 41.32 ms` in the slice, `12.299 → 12.038 ms/token`.

**This is the fourth time in one run that a correct rejection went stale**, and it is the most
transferable observation in this document:

| | rejection | what invalidated it |
|---|---|---|
| O4m | QKV grid capped at 32 | fusing Q/K/V widened N |
| O4k | fused K/V write unavailable | (V relocation made it legal) |
| O4o→**O4p** | gate/up plan "costs more than it buys" | the norm now emits the shard for free |
| hand-port `[gpt-28]` | sharded norm, +4.4 ms worse (§6.39/§6.40) | §6.65 traced the reshard dispatch away → §6.67 reversed it |

**A rejection records a measurement under conditions, not a fact.** Every structural change silently
re-prices every knob and every earlier "no". The tool's real advantage over a human pass is not that
it finds cleverer optimizations — five of its wins the hand-port already had, and two of them the
hand-port does better (O4n, and the decode-mode RoPE). **It is that it keeps re-opening its own
closed questions, cheaply, in an order driven by fresh measurements.** A human does that once or
twice, when something prompts it; §6.67 and §6.72 are the hand-port's two.

**For the hand-port this is the standing recommendation behind items #2, #4, #5 and #6:** the
rejections in `STATUS.md` are dated, and the structure has moved underneath several of them.

### O4q — the shard chain, extended to the whole block

The last of the layout-chaining family, and the one that shows how far it goes:

> *The stream is one tile row of 3072 values that every op in the block already touches in L1, but
> it went back to interleaved between them, so each norm re-opened the same shard from DRAM — twice
> per layer, 26 layers deep. [...] Both norms in a block are built on the same dim, and so is the
> next layer's, so **the stream can stay in the shard the whole way down**.*

`12.038 → 11.928 ms/token`, then a further `→ 11.839` closing the loop from the other end — `o_proj`
and `down_proj` now emit the residual add's shard directly, so the stream is sharded end to end
through the block rather than only on the norm side. Folded into take-item #4 rather than listed separately, because for the
hand-port it is the same recommendation carried further: not just norm→QKV, but the residual stream
never leaving its shard across the depth of the model.

Worth noting the hand-port already solved the *adjacent* problem better — `[gpt-27]` passes the
residual as the matmul's `bias`, so the add itself costs no launch at all. That is orthogonal to
this and the two compose.

### O8 — the accept test has no exchange rate, and it shows at the end of a run

The last commit bought **0.012 ms/token (0.10%)** and cost **0.0129 of PCC** — 0.9903 → 0.9774, about
a quarter of the remaining headroom above the 0.95 gate, for a gain indistinguishable from noise.

Nothing in the ladder objects, because the accept test is:

```
faster?  AND  PCC still above the floor?   ->  keep
```

There is no notion of an **exchange rate** — no "is this gain worth this much accuracy?" So once the
structural ideas run out, a run will keep converting PCC headroom into arbitrarily small wins until
it reaches 0.95. The run's own arc shows the transition cleanly: twelve consecutive layout and
dispatch wins at **bit-identical PCC**, and then this.

**Suggested fix, and it is small:** require a precision-spending change to clear a ratio, not just
the floor — e.g. reject when `Δpcc / Δms` is worse than some rate, or simply require dtype-rung
wins to exceed a materiality threshold (the tool already has `material_gap_threshold_ms = 0.25`
for choosing targets; the same idea applied to *accepting* precision trades would have rejected
this). Cheap to add, and it is the difference between a run that stops at its best result and one
that grinds its accuracy down for noise.

**This is the mechanised form of §6.16.** The hand-port faced the identical question, computed the
exchange rate explicitly — *w2 costs 77% of the precision stack's accuracy for 15% of its speed* —
and handed back 2.5 ms. That reasoning has no place to live in the current design. Together with O1
(no per-weight dtype axis) this is the clearest gap between the tool and a careful human pass.

### F13 — the generated stubs swallow fast-path exceptions, so a perf regression passes the PCC gate

**Status: found by the agent itself, 2026-08-14** · severity: a broken optimization reports as
correct · reported: not yet

The emitted `tt/pipeline.py` guards its fast paths like this:

```python
try:
    ...fast matmul plan...
except Exception:
    self.lm_head_dram = None      # then fall through to plain ttnn.linear
```

The fallback computes **the same math**, so `check_pcc` passes. What is lost is only speed — and
PCC cannot see a performance-only failure. The guard converts an exception into a **silent policy
change** rather than an error.

**Measured instance.** Setting the LM head's output dtype to `bfloat4_b` made
`ttnn.to_layout(part, ROW_MAJOR_LAYOUT)` throw — block-float formats require TILE layout, and while
`bfloat8_b` is accepted, `bfloat4_b` is not. PCC still reported **ok at 0.9774**, while all 128
decode tokens ran the interleaved fallback: **device_ms 223.65 → 244.64**, and the op table showed
`32 x 3072 x 131072` at n=128 where `32 x 3072 x 32768` at n=512 should have been.

**Why it matters here specifically:** this tool's entire design premise is that *the AI proposes and
the harness verifies*. A guard that turns a failed optimization into a passing test defeats that
premise inside the generated code itself — the harness is verifying honestly, but it is being handed
a program that lies about which path it took.

**Fix, two options:** (a) do not emit bare `except Exception` around fast paths — let the failure be
loud during bring-up, and gate the fallback on an explicit capability check instead; or (b) if the
guard must stay (the RoPE commit's latched fallback exists for a real reason — an exception inside a
captured trace is fatal), then have `measure_candidate` assert the **expected op signature and call
count** are present in the profile, not just that the run was faster. The tool already parses that
table.

**Credit where due: the agent caught this itself**, diagnosed it from a collapsed call count in the
per-op table, recorded the general rule (*"never trust `check_pcc` alone after touching a guarded
fast path"*), and settled on `bfloat8_b`, which does not trip the guard.

### O5 — the ladder's escalation is real, and it gives up in the right place

On the LM head matmul it climbed `grid → dtype → shard → fidelity → structural → cpp` across 11
attempts, and the C++ rung came back *slower* (400.19 vs 397.27) and was reverted. It then moved on
rather than grinding. The rung ordering by `bound_by` behaved as documented.

### O6 — `weight_dtype` reads `null`, so the dtype rung is recommended blind

Every entry in `blocking_ops` reports `"weight_dtype": null` and the advice text renders as *"lower
the weight dtype (now unknown) to bf8_b/bf4_b"*. The profiler is not recovering the dtype actually
in use, so the tool cannot tell an untouched bf16 weight from one it already stepped down, and will
re-propose the same rung on a weight that is already at bf4_b. Minor — the agent notices from its
own diff — but it wastes attempts and weakens the report. **Worth fixing.**

### O7 — the throughput band can never stop the run

`termination_check` carries `"band_stop_disarmed": "divisor is an estimate (device census: 7.18 GB
resident at served dtype), not an exact param count"`. Reaching `IN_BAND` therefore does not stop
the loop, by design. Defensible — an estimated divisor should not end a run — but it means the
documented "stops when at the theoretical floor" exit is unreachable for any model whose parameter
count is inferred rather than declared, which is every model that arrives as a local directory.

---

### Blindness audit — evidence that the comparison is honest

The whole experiment is worthless if the tool saw the hand-port. Audited 2026-08-13 rather than
assumed. Method: grep the agent's full 6.7 MB tool transcript, and enumerate **every** file-access
tool call it made.

| check | result |
|---|---|
| `models/experimental/voxtral_tts` in the tool's checkout | absent (not in the working tree) |
| transcript hits: `repos/tt-metal` outside `-pr46283` | **0** |
| transcript hits: `experimental/voxtral_tts`, `ttnn_voxtral`, `STATUS.md`, `ONBOARDING.md` | **0** |
| transcript hits: the hand-port's headline numbers (`45.4`, `26.9`, `RTF`) | **0** |
| `WebSearch` / `WebFetch` calls | **0** |
| every `Read`/`Glob`/`Grep` target | all inside `pr46283`, except two memory files |

The two exceptions are `.claude/projects/-localdev-lserbedzija-repos-tt-metal/memory/`
`python-env-wrapper-fixes-planner-gates.md` and `perf-mcp-env-traps-look-like-device-faults.md`
(the latter written by the agent itself). That scope holds four memories, all created *during this
experiment* and all about the tool's own environment traps — `python_env` wrappers, `PYTHONPATH`,
the `ttnn` namespace-package trap. No optimization content. The hand-port's own memories are in a
different scope (`-localdev-lserbedzija/memory/`) which appears nowhere in the transcript.

Eight matches for a `6.6x` pattern were checked individually and are all coincidental decimals
(`316.6906`, timestamp fractions), not `STATUS.md` section references.

**Two honest qualifications.** (1) Both branches share the `tt-metal` remote, so the hand-port is
reachable from inside this checkout via `git show <sha>:<path>` even though it is not checked out —
it was never impossible to reach, it simply was never reached, and the transcript is the evidence.
(2) The real contamination vector was the operator, not the tool: what was handed to it was
tool-source fixes (F1/F2/F5), `conftest.py`, env wrappers, and the HF export. None encodes tuning,
and the tool justified its own wins from primary sources — the argmax finding cites
`argmax_device_operation.cpp:16` by line, not a measurement from the hand-port.

A write/read tripwire on the hand-port tree (1,424 files fingerprinted at `fa57362fe5`, clean) ran
for the remainder of the experiment.

---

## ★ TRANSFER RESULTS — what happened when the tool's findings were applied to the hand-port

The reason the experiment existed. Every precision-neutral finding was implemented against the
hand-port and measured on the same p150, with its own audio-tier gate (45 utterances, WER, MOS,
6 PCCs, ms_per_frame) run twice in one session so nothing is judged against a stored number.

Baseline, reproducing §6.71 to the noise floor: **ms_per_frame 27.751, RTF 0.3653, WER 0/894,
MOS 4.6101, 132/132 tests, 32 metrics with no nulls.**

| # | tool finding | outcome on the hand-port |
|---|---|---|
| O4o/O4q | residual stream kept in the norm's shard | **−0.99 ms/frame** — and **WER 0 → 2**, fails the gate |
| O4f | Q+K rotated in one call | **refused** — `nlp_create_qkv_heads_decode` returns q/k already `HEIGHT_SHARDED`; concat-then-rotate is a TT_FATAL |
| O4k | fused K/V cache write | **0.510 ms/token SLOWER** — §6.44 independently reproduced |
| — | SwiGLU product → `w2` as a sharded operand | **refused** — matmul rejects a sharded in0 |
| — | head-merge → `wo` as a sharded operand | **accepted, but slower AND wrong** (PCC 0.954, `nan` at 16 cores) |
| — | reshard the QKV activation once | **moot** — the hand-port already fuses q/k/v into one projection |
| O4b | per-op grid for `w2` | **~0.15 ms, under the instrument's resolution** — §6.52 stands |

**Nine further findings were already in the hand-port** (§6.67 sharded norm, §6.65 traced decode,
§6.52 program configs, fused `wqkv`, fused RoPE, `[gpt-05]` decode-native layout — worth **6.6
ms/frame** there against the tool's 0.93 — `[gpt-21]` SDPA core count, §6.72 head split, and
`[gpt-26]` `fused_activation`, which the hand-port does *better*, see F12).

### The headline, stated plainly

**Essentially nothing from this tool transfers to a mature hand-port of the same model.** That is
not a criticism of the tool — it is a statement about where its value lies. Its wins were real and
large *on its own output*: its `w2` sat at 96 cores, its RoPE was a seven-op chain, its Q/K/V wore a
prefill layout, its argmax was single-cored. The hand-port had already fixed all of those. What the
tool recovers is the distance between generated code and hand-tuned code — and by construction that
distance is zero once someone has done the hand-tuning.

### The one thing that DID transfer, and why it still failed

O4o/O4q is a genuine **0.99 ms/frame** — double the gate's 0.5 ms tolerance. It failed on WER
because the chain forced `_NORM_GRID` from 32 to 48 cores, which changes the RMSNorm's reduction
tree (48 partials, not 32). Not bit-identical, `decode_min_pcc` 0.999316 → 0.999288, and 2 of 894
long-form words flipped. **A variant pinning `per_core_N=3` on the two residual matmuls — which puts
their output on 96/3 = 32 cores, matching the norm's existing grid — is under test.**

### ★ O9 — what the 0.95 gate actually cost, measured in the metrics it cannot see

The tool's optimizer accepted `down_proj → bfloat8_b` (commit `074ec705a8`) on the only two
criteria it has: faster, and PCC still above 0.95. The hand-port evaluated the same change at
§6.16 and **handed the speed back**. This prices that disagreement by making the identical change
to the hand-port and running its audio-tier gate — everything else held fixed, both tags in one
session.

```
                    baseline      w2 -> bf8_b
wer_longform             0    ->      3        *** WORSE ***   (tolerance is 0)
codes_real_n            45    ->     59        *** WORSE ***   +31% code flips
codes_real_pct         5.2    ->    6.8        *** WORSE ***
decode_mean_pp        0.97    ->   1.09        *** WORSE ***
mos_min             2.6597    -> 2.4145                        -0.245 on the WORST utterance
mos_longform        4.6101    -> 4.5895                        within tolerance
ms_per_frame        27.751    -> 30.271        *** WORSE ***   SLOWER by 2.52 ms
rtf                 0.3653    -> 0.4113        *** WORSE ***
pytest                 132    ->    131        *** WORSE ***   the §6.16 guard fired
                                               8 metrics worse, 15 within tolerance
```

**Three of those four quality columns are invisible to the tool's gate.** It has no word-error
metric, no perceptual metric, and no exact-match check on discrete codes — and `mos_min` is the
worst-case utterance, which a mean-based gate would hide even if it had one. Its own PCC reading
for this change stayed comfortably above 0.95 throughout.

**And on this build the change is SLOWER.** §6.16 measured BFP8 `w2` as ~2.5 ms *faster* on the
N150; on p150 it costs 2.52 ms. So the trade is now loss-loss.

#### O9b — the tool re-opens its NOs but never its YESes

The sharpest part. In run 2 the tool's own diagnostic for this exact op reported:

> *this op takes the SAME 0.159 ms/call at bf16 and at bf8_b, so it is not bandwidth-bound and the
> limit is the shape of its k-reduction*

**It measured that `w2`'s dtype does not affect its speed — after having already spent accuracy
lowering that dtype in run 1 — and never went back.** Section O4p credits this tool for re-opening
four *rejections* in a single run, which is the best thing it does. But nothing re-opens an
**acceptance**. Precision is the only irreversible cost the ladder pays, and it is the one class of
decision never revisited when later evidence undermines it.

**Suggested fix, and it is symmetric with what the tool already does well:** when a measurement
shows a lever is inert on an op (same time at two dtypes, same time at two grids), re-open every
*applied* change that rung made to that op and try reverting it. A revert that costs nothing in
time and buys back accuracy is a strict win, and the ladder currently cannot express it.

### F14 — "producer emits the consumer's shard" needs to check the consumer's PROGRAM CONFIG grid

The most useful mechanism the tool found is also the one most likely to misfire, and the reason is
worth handing to the PR author directly:

> `memory_config=` on a matmul carrying a program config is **only a request for the layout**. The
> shard spec follows the matmul's OWN grid. Asking `DECODE_PRG["wo"]` (`_MM_GRID=(12,6)`,
> `per_core_N=2`) for a 32-core shard returns a **48-core** one, and the downstream norm refuses it:
> *"shard_spec.grid size 12x4 does not fit within program_config grid 8x4"*.

So a tool applying this lever has three choices, and it should know which it is making: move the
consumer to the producer's grid (what the tool did — and what perturbs a reduction's arithmetic),
move the producer to the consumer's (`per_core_N`, arithmetic-preserving), or give up. **The tool
took the first without recording that it had changed the numerics** — its PCC gate at 0.95 could not
see a 2-word WER shift, and nothing in the ladder flags "this lever altered a reduction tree".

**Suggested fix:** when a shard-chaining lever changes a *reduction's* core count (norm, softmax,
argmax, any tree reduce), record it as a precision-affecting change rather than a layout one, and
prefer the `per_core_N` route when it exists.

### Process note — a measurement error of mine, and how it was caught

My first fused-K/V-write measurement showed it **8.72 µs faster** and I nearly reported §6.44 as
overturned. The V relocation was hoisted **out** of the timed region, so it measured the fused write
as if V were already on a disjoint core — which it never is, because head creation puts K and V on
the same core at batch 1. With the move inside the loop the same comparison reads **19.63 µs
slower**, and the V move alone costs 23.49 µs. §6.44 recorded 0.687 ms/step; the corrected
measurement says 0.510 ms/token. Same sign, same magnitude, independently reproduced.

The general form: **when testing a fused op that requires an operand to be relocated, the relocation
is part of the cost unless the producer can emit it in place.**

---

## ★★ WHAT THE OPTIMIZER IS MISSING — the analysis this experiment was for

The ladder, as defined in `run.py:102`:

```
knob:grid -> knob:fidelity -> knob:dtype -> knob:shard -> structural -> tt-lang -> cpp
                                            structural levers named: trace | kv-cache | gather
```

### 1. Coverage map — the hand-port's shipped wins against the rung that would find them

| hand-port win | worth | rung that finds it |
|---|---|---|
| sharded decode RMSNorm (§6.67) | **−5.399 ms/frame** | `knob:shard` ✅ found it |
| decode matmul program configs (§6.52) | **−5.06** | `knob:grid` ✅ — but the silu half is **unreachable**, see F12 |
| whole frame graph traced (§6.65) | **−4.244** | `structural:trace` ✅ found it |
| sdpa for Block 2's attention interior (§6.45) | −2.555 | ❌ **no rung** — swap a hand-rolled interior for a library primitive |
| residual as matmul bias (§6.62) | −1.918/step | ❌ **no rung** — algebraic rewrite |
| in-place elementwise, Block 1 (§6.47) | +0.929 | ❌ **no rung** — allocation elimination |
| two plain KV writes + 1-core qkv shard (§6.44) | +0.907 | ⚠ `knob:shard` in reverse — the rung only ever ADDS sharding |
| in-place elementwise, Block 2 (§6.48) | +0.790 | ❌ **no rung** |
| hand-rolled 9-op head split (§6.72) | −0.775 bit-exact | ❌ **no rung** — this is DE-fusion |
| `_SDPA_PRG` (§6.46) | +0.197 | `knob:grid` ✅ |

Plus everything inherited and still shipping — **CFG batch-fold into rows (2.23×)**, qkv weight
fusion, `SCALE` baked into wqkv's q rows, `_trunk` projecting before it narrows, the semantic argmax
**on the host**, the codec's gather-based pad. **None of those has a rung either.**

**By magnitude the ladder reaches about two-thirds of the device-time wins and none of the algebraic
ones.** And that is the generous reading — it assumes `knob:grid` reaches `fused_activation`, which
F12 shows it does not.

### 2. The real finding: `structural` is where the value was, and it is the least specified rung

Every large win the tool itself landed in run 2 came from `structural` — fused QKV, the decode-native
layout, one-call RoPE, the shard chain. Yet the rung names only **three** levers (trace, kv-cache,
gather), none of which is any of those. **The agent improvised all of it.** That is why the run was
good and also why it is not reproducible: the ladder's most valuable rung is a blank cheque.

**Recommendation: populate `structural` with a named sub-catalogue**, each with a firing condition
and a guard. Every one below is drawn from measured evidence, with both signs where they exist:

| sub-lever | fires when | evidence |
|---|---|---|
| **`bias-fold`** | an elementwise add's only consumer is a matmul → make it that matmul's `bias` | §6.62, **−1.918 ms/step**. Guard: one tile of rows only — a bias broadcasts and is silently wrong on prefill |
| **`in-place`** | an elementwise operand is dead immediately after → use the `_` variant | §6.47 + §6.48, **+1.72 ms** combined; allocation was ~12 µs of a ~65 µs op |
| **`reorder project↔narrow`** | a projection is adjacent to a slice/gather/duplicate → try BOTH orders | `_trunk` projects before narrowing (**win**); §6.34 project-then-duplicate is **0.785×, a loss**. Both signs — must be measured, never assumed |
| **`weight-bake`** | a constant scalar multiplies a projection's output → fold it into the weights at load | `SCALE` into wqkv's q rows |
| **`weight-concat`** | sibling projections consume the same activation → concatenate at load | the tool DID find this (O4c); worth naming so it is not re-derived |
| **`de-fuse`** | a library op can be expressed as primitives, **and trace is applied** | §6.72, **−0.775 ms bit-exact**, 9 ops beating 1. The tool found this too (O4e) — from the opposite direction, and only by accident |
| **`library-swap`** | a hand-rolled interior matches a library primitive's contract | §6.45 sdpa, **−2.555 ms** |
| **`revert`** | a previously-applied config → try REMOVING it | §6.43: `wo`'s tuned config was inert, and deleting it was bit-exact |

### 3. Three structural blindnesses — things the design cannot express

**(a) The host is forbidden as a destination.** `test_e2e_pipeline` asserts `torch_ops == 0` and
`test_forward_fires_no_host_op` asserts zero host aten ops. So "this work belongs on the host" is
**inexpressible**. §6.8 moved a semantic argmax to the host for **1.439×** — an 8320-value reduce
that already ended in a D→H copy, so it added no round trip. §6.50 is the control: moving the other
three host steps ON device is 7–29× slower. A tool that can only ever move work onto the chip will
never find either. **Suggested fix:** allow a host fallback when the op is already adjacent to a
transfer, and gate it on total wall time rather than on op location.

**(b) Op count is treated as monotone-good.** Every lever reduces launches. §6.72 and the tool's own
O4e both show the reverse winning once trace has removed launch cost — *"dispatches fell 3413 → 2867
yet it got slower; these were view ops doing no work."* **De-fusion should be a scheduled rung after
`structural:trace`, not a lucky accident.**

**(c) PCC at one length is the whole correctness model.** `[gpt-21]` records SDPA settings that were
faster and **"NOT SAFE — position sweep"**; §6.31 holds back a 2.079× bf16 semantic head because
*"one flip redirects the whole utterance"*. Neither is visible to a single-length PCC gate. **A
generative model needs its gate run at several positions/lengths**, and discrete outputs (argmax,
codes) need an exact-match check, not a correlation.

### 4. Already filed, restated here because they belong to this analysis

- **O1** — no per-weight dtype axis. §6.16 kept `w2` in bf16 while everything else went BFP8, because
  w2 alone was 77% of the precision stack's accuracy cost for 15% of its speed.
- **O8** — the accept test has no exchange rate: `faster AND above the floor` keeps a 0.10% gain that
  costs a quarter of the PCC headroom.
- **F12** — the fusion rung reaches for a grid when it should emit a full program config, so
  activation fusion is recorded as a loss when the lever was never pulled.
- **F14** — shard-chaining must check the consumer's program-config grid, and must classify a change
  to a REDUCTION's core count as precision-affecting.

### 5. What to keep — the tool's genuine structural advantage

Worth saying plainly, because the rest of this section is criticism. **The tool re-opened its own
closed questions four times in a single run** (O4m, O4k, O4o→O4p, plus the stale 32-core cap) and
that is where its best late wins came from. The hand-port did it twice in 74 experiments (§6.67,
§6.72) and its own ledger calls the rule out: *"a rejection is stale when its premise is a cost
someone has since removed."*

**That behaviour should be promoted from emergent to designed:** when any structural change lands,
mark every knob measured before it as stale and re-open it. The tool nearly has this already — it is
the single thing it does better than a careful human, and it is currently an accident of the agent's
judgement rather than a property of the ladder.

---

## ★★ THE THREE-BLOCK EXPERIMENT — packaging Voxtral-TTS as one HF model

The Block-1-only run was criticised, correctly, for testing the model as a text LM when the
deployment drives it with audio. So the whole pipeline was packaged as a single
`trust_remote_code` HuggingFace model and handed to the tool. **It works**: 4.00 B parameters,
three blocks, bit-exact to the reference (`torch.equal`, maxdiff 0.0 on all three), self-contained,
text ids -> 24 kHz audio in one `forward`.

### S4 — six packaging defects, all in `transformers`, all hit in one afternoon (OURS/upstream)

Not tool defects, but every one is a barrier between a real research model and this tool, and the
tool's adoption depends on people clearing them:

| # | what | how it fails |
|---|---|---|
| 1 | `save_pretrained` drops `auto_map` unless `register_for_auto_class()` was called | later load says *"Transformers does not recognize this architecture"* — blames the architecture, not the missing key |
| 2 | `trust_remote_code` does not support **subpackages** | `from .reference import x` is resolved as a file `reference.py`; all custom code must be flat |
| 3 | its import scanner only matches `from .module import name` | the `from . import module` form is **invisible**, so those files are never copied and fail at runtime |
| 4 | only `.py` files are copied to the module cache | any asset resolved from `__file__` (voice presets, `params.json`, vocab) breaks, because `__file__` is now the cache |
| 5 | `ModelOutput` subclasses need `@dataclass` | fires on the **return** path, after the entire forward has already run |
| 6 | `nn.ParameterDict` forbids `.` in keys | every real checkpoint uses dotted names |

### F15 — `plan` and `compat` disagree about what the model IS

Same model, same directory, two stages of the same tool, run minutes apart:

```
plan   :  Category: TTS  (pipeline_tag=None, library=transformers)
          Category guidance (TTS): Text-to-speech. Closest template: models/demos/qwen3_tts/
          CONFIDENCE: LOW

compat :  Architecture: unknown / non-LLM (fingerprint: unknown)
          Overall verdict: ARCHITECTURE NOT RECOGNIZED (non-LLM) — no confident block plan
          Summary: 0 ready / 0 partial / 0 missing
```

`plan` identified it as TTS **from the config alone** (`pipeline_tag=None`, so not from Hub
metadata) and even named a TTS template it could copy. `compat` then declared the architecture
unrecognised and emitted an **empty** block table. A user reading these in order gets a green light
and then a wall, with no explanation of which stage to believe.

**Credit where due:** `plan`'s `CONFIDENCE: LOW` here is *correct* and is an improvement on the
Block-1 run, which printed `CONFIDENCE: HIGH` while admitting it had omitted the KV term (O3).

### F16 — the block table degrades to EMPTY, which reads identically to "nothing needed"

Block 1 alone produced `11 ready / 0 partial / 0 missing`. The three-block model produces
`0 ready / 0 partial / 0 missing` — not "these blocks are missing", but **no analysis at all**,
printed in the same format. Combined with **F2** (the READY verdict can never fire because the
predicate is `lambda _: []`), the summary line is now unable to express either success or failure
for a custom architecture: 0/0/0 is what you get whether everything is supported or nothing was
examined.

**Suggested fix:** distinguish *not analysed* from *analysed, nothing missing*. A third state, or
simply refusing to print the summary line when the analyser bailed.

### F17 — machine-readable structure is declared and never read

The config states the model's shape in fields designed to be read:

```json
"task": "text-to-speech",
"block_stacks": ["backbone", "flow", "codec"],
"decode_input": "audio_code_embedding"
```

`plan` used `task`. **Nothing used `block_stacks`.** `compat`'s advice is
*"inspect subfolders (dit/, vae/, text_encoder/, ...) and bring up per-component"* — the Stable
Diffusion **folder** convention. This model's three stacks are `nn.Module` attributes, not
subfolders, and are named in the config. The tool has multi-stack support (its own commits
`emit-e2e: a multi-stack model must expose one depth knob per stack` and `G6 refuses a model whose
block stacks the profiler cannot see`), but discovery is folder-shaped only.

**Suggested fix, and it is small:** when the config declares stacks, walk those attributes. A model
that says what it is should not have to also be laid out in a particular directory shape.

### What still works on an unrecognised architecture — worth keeping

`compat` did not simply bail. It read the real config and produced genuine kernel-level findings:

- `ttnn.topk (sampling)` — `vocab_size=131072` needs a power of two **< 65536** for the multi-core
  path, else single-core, with the throughput consequence stated
- per-TP divisibility, correctly deriving that `TP=32` fails on `num_key_value_heads=8` while
  TP=1/2/4/8 are fine, and correctly framing it as *"rules out that mesh shape, not the model"*

So the kernel-constraint half degrades gracefully where the architecture half does not.

---

## ★ F18 — the architecture gate tests the model's NAME, not its structure (with a tested fix)

`compat` refused the three-block model outright. The chain, in `compatibility.py`:

```python
family = detect_family(cfg)              # matches cfg["model_type"] against hardcoded name lists
is_unknown = family.startswith("unknown")
fpr = arch_descriptor(model_type, architectures, is_encoder_decoder)
if not fpr.startswith("decoder-only"):
    return report                        # early return -> EMPTY block table
```

**Every input to that decision is a name.** `model_type`, `architectures`, `is_encoder_decoder`.
`model_type` is free text chosen by whoever wrote the config, so a model that is *structurally* a
Llama-family decoder is refused for being called `"voxtral_tts"` — while the config carries every
field the block checks actually read: `num_attention_heads`, `num_key_value_heads`, `head_dim`,
`intermediate_size`, `rope_theta`, `rms_norm_eps`.

**The intent is right and worth preserving.** The checklist it runs is the LLM-decoder one; run it
against a VAE and it would report *"GQA attention: ready"* for a model with no attention. Families
genuinely differ — MLA, SSM and MoE need different handling. Refusing to guess beats a confident
plan for the wrong architecture.

**The implementation tests the label instead of the thing.**

### Tested fix

Added `_looks_like_decoder(cfg)` — requires the four fields the checks read, plus one
attention-shape hint (`num_key_value_heads` or `head_dim`) and one position hint (`rope_theta`,
`rope_parameters`, `rope_scaling` or `max_position_embeddings`), so a bag of integers does not
qualify. Used as a fallback in `detect_family` after the name lists.

Same model, same command:

```
BEFORE   ARCHITECTURE NOT RECOGNIZED (non-LLM)      0 ready / 0 partial / 0 missing
AFTER    Llama-family causal LM (INFERRED from config fields, model_type='voxtral_tts'
         is not a known name; config declares 3 stacks: backbone, flow, codec)
         Overall verdict: READY                     10 ready / 0 partial / 0 missing
```

Also added `declared_stacks(cfg)`, which reads `block_stacks` — the field F17 showed nothing was
reading.

### The fix is INCOMPLETE, and the remainder is the real recommendation

`Overall verdict: READY` is **wrong**. It analysed the *backbone* and declared the whole *model*
ready. The flow matcher and the codec are not covered by that checklist at all, and the codec is
built on `conv1d` — which §6.13 records as the op that caused this port's hang. So the patch trades
"refuses to analyse anything" for "analyses one stack and overclaims for three": **F16 in a new
costume.**

The correct output is per-stack, and the tool already has the names:

```
backbone : 10 ready / 0 partial / 0 missing
flow     : NOT ANALYSED — no checklist for flow-matching
codec    : NOT ANALYSED — no checklist for neural codecs
```

**Three states, not two: supported / missing / not-analysed.** That is the change worth making —
it is what lets the tool be honest about scope instead of choosing between silence and
over-confidence, and it is the same gap F16 identified from the other direction.

*(Note: the `READY` verdict can only fire at all because F2 was fixed in this checkout —
`lambda _: []` -> `lambda _: True`. Unpatched, that verdict is unreachable.)*

---

## ★★ F19 — template dispatch silently runs a DIFFERENT model, and the template can be the tool's own earlier output

The highest-severity finding of the three-block experiment, because it produces a complete,
plausible-looking run whose results are not the model you asked for.

Handed `/localdev/.../voxtral-tts-full` (three blocks, 4.00 B params), `auto-up` printed:

```
  Step 2/6  Scaffold the demo folder for /localdev/.../voxtral-tts-full
  GENERIC LLM BACKEND. No per-model tt/ folder needed. Skipping scaffold and
  routing directly to `prepare --execute`.
  ALREADY SUPPORTED via tt_transformers/simple_text_demo. Skipping scaffold.

  BRING-UP TEMPLATE — /localdev/.../voxtral-tts-full on P150 mesh [1,1]
  Backend: Voxtral TTS Backbone (mistral decoder)
  Runs canonical HF id out-of-the-box: /localdev/.../voxtral-tts-backbone     <-- DIFFERENT MODEL
  Compat verdict: READY
```

It dispatched to `models/demos/voxtral_tts_backbone/`, whose demo loads a **different checkpoint
directory** — the Block-1-only export from the previous experiment. Left overnight this yields a
finished run, with metrics, for a model nobody asked about.

**Three things make this dangerous rather than merely wrong:**

1. **The warning is buried.** It exists — *"adapt encoder/decoder/IO ... before expecting correct
   outputs"* — as a prose bullet under `Notes:`, below a header stating the other model's id as a
   feature (*"Runs canonical HF id out-of-the-box"*) and directly beneath `Compat verdict: READY`.
2. **The template is the tool's OWN prior output.** `models/demos/voxtral_tts_backbone/` was
   generated by an earlier run of this same tool. It found its own artifact and reused it as a
   template for a different model. Nothing checks that a template's `canonical_hf_id` is the model
   under port.
3. **Skipping scaffold is silent about consequence.** "No per-model tt/ folder needed" reads as an
   optimisation. What it means is that no port happens at all.

**Suggested fixes, cheapest first:**

- **Refuse, do not warn, when `template.canonical_hf_id != model_id`** unless an explicit
  `--allow-template-substitution` is passed. A different checkpoint is not a detail to note.
- Never treat a directory the tool generated as a template for a *different* model id.
- Say what skipping scaffold costs: *"no TTNN port will be produced for this model"*.

### F18 (correction) — my own fix caused F19 to fire, and that is the sharper finding

The routing gate is:

```python
if compat.overall == "READY" and not _missing and not _partial and _generic_backend_picked:
    _route_via_generic_llm = True          # -> skip scaffold, run a sibling demo
```

F18's patch made `compat` return `READY` for a family inferred from config structure. That flipped
this gate. **So the patch did not merely overclaim in a report — it caused the port to be skipped
entirely.** Recorded against myself because it is the clearest possible demonstration of the
underlying design problem: `READY` is doing two incompatible jobs, "the checklist passes" and
"this model needs no work", and any change to the first silently changes the second.

**Corrected patch.** Generic routing now additionally requires that the family came from a known
`model_type` **and** that the config declares at most one stack, and it states why when it declines:

```
NOT routing to the generic LLM demo: family was INFERRED from config structure, not a
known model_type; config declares 3 stacks (backbone, flow, codec) and the block
checklist covers the decoder stack only. Scaffolding this model's own stubs instead.
```

`CompatReport` now carries `family_inferred` and `declared_stacks` so the caller can tell an
inferred verdict from a matched one — which is the general form of the fix: **a verdict should
carry how confident it is and what it covered, not just what it concluded.**

---

## ★★★ F20 — the meta-plan already knows. It is wired to stdout, not to control flow.

**This is the most actionable finding in this document, and the cheapest to fix, because the
analysis already exists and is good.**

Before failing, `auto-up`'s advisory meta-plan wrote this about the three-block model — unprompted,
with no access to anything in this file:

> *This is a three-stage TTS pipeline (an LLM-style 26-layer backbone, an audio codec decoder, and
> a flow-matching vocoder) being routed into `tt_transformers/simple_text_demo` purely by
> category-default*

Its six listed risks are, one for one, the findings the rest of this document arrived at the hard
way:

| the meta-plan said | this document filed it as |
|---|---|
| *"Backend selection is a category-default match, not a genuine 'voxtral_tts' match — simple_text_demo assumes causal text-token generation"* | **F19** (template dispatch runs a different model) |
| *"Early success graduating the 26-layer backbone (the 'easy' REUSE/ADAPT part) may create false confidence that masks the fact that the two hard components are architecturally out of scope"* | **F16 / F18** (READY overclaims scope) — stated better than I stated it |
| *"Do not evaluate codec_decoder/flow_matching against simple_text_demo's logit-based PCC harness — they need a waveform/mel-level comparison harness, which doesn't exist in this backend"* | the **whole correctness-gate argument** (O9, §3 of the optimizer analysis) |
| *"Both audio-specific components show leaves=1, meaning the discovery tracer could not see inside them"* | **F17** (structure declared, not read) |
| *"flow-matching ... a multi-step numerical integration loop, not a fixed op graph — which the op catalog has no [coverage for]"* | the `structural` rung's missing sub-levers |
| *"variable-step sampling overlaps with the already-flagged unsupported 'DynamicShape' NEW-op category"* | — |

It then recommended, correctly:

> *"Cap auto-iterate retry budget on codec_decoder and flow_matching specifically and escalate to
> human review early, rather than letting the loop retry against op patterns that were never in its
> catalog."*

And immediately printed:

```
(advisory only; proceeding with auto-iterate loop. Disable via --no-meta-plan.)
```

**It identified two components as architecturally out of scope, recommended capping their retry
budget and escalating to a human, and then proceeded to do exactly what it had just warned
against.** The run subsequently died on an iteration-budget timeout — spent on the components the
meta-plan had named half an hour earlier.

### Why this reframes most of this document

The gap in this tool is **not analysis**. A component of it already produces a better architectural
critique than the rest of the pipeline acts on. Every finding above — the name-based family gate,
the overloaded `READY`, the template substitution, the logit-only correctness harness — is visible
to the meta-plan and invisible to the code that makes decisions.

### Suggested fix, in increasing order of ambition

1. **Let the meta-plan set budgets.** It already emits per-component risk. Feeding
   `cap_iterations(component, n)` and `escalate_to_human(component)` back into the loop is a
   plumbing change, not a research one — and it is the tool's own recommendation.
2. **Let it veto a backend.** When it says *"backend selection is a category-default match, not a
   genuine match"*, that is F19's check already written in prose. Make it a refusal.
3. **Let it select the correctness harness.** *"they need a waveform/mel-level comparison harness"*
   is the gate-design decision the optimizer half needs (O9). The meta-plan knows which harness a
   component requires; nothing asks it.

**Until then the tool prints an accurate diagnosis and then ignores it, which is a worse user
experience than not producing the diagnosis at all** — the run looks informed and behaves as though
it is not.

*(Recorded verbatim in `autoup_full2.log` lines 175-193.)*

### ⚠ F20 REVISED — the pipeline ignored the meta-plan and was RIGHT

The framing above is too strong, and the run that finally completed refutes it. The meta-plan
called `voxtral_flow_matching` and `voxtral_codec_decoder` *"architecturally out of scope for this
backend"* and recommended capping their retry budget and escalating to a human. The loop ignored
that and **ported both**:

```
23:03  ✓ GRADUATED  voxtral_codec_decoder   5/7
29:57  ✓ GRADUATED  voxtral_flow_matching   6/7
Graduated (ON_DEVICE): 7/7 (100%) actually graduated (native stub, PCC-verified)
```

Verified as real TTNN, not a torch shim that would pass a PCC gate trivially: `voxtral_codec_decoder.py`
is 363 lines with 11 `ttnn.linear`, 6 `ttnn.slice`, 5 `ttnn.reshape/multiply/add`, 4
`ttnn.permute/concat`, 2 `ttnn.matmul/embedding` and `ttnn.transformer`, **no `except`/fallback
path**, and `from_torch` only for weight upload.

**So the corrected recommendation is narrower and better:** the meta-plan's architectural pessimism
should inform **budgets and ordering** — try the hard components last, cap their share of the
iteration budget, warn the user what is at risk — but it must **not veto attempts**. It was wrong
about both hard components, and a veto would have cost the run's most valuable result.

What survives from F20 unchanged: its *factual* observations were all correct (the category-default
backend match, `leaves=1` discovery blindness, the logit-only PCC harness being wrong for a codec
and a vocoder), and none of those is wired to anything. **Route the facts into control flow; leave
the predictions advisory.**

---

## ★★★ THE OVERNIGHT RESULT — full three-block port, 7/7, 34 minutes

The experiment the whole detour was for.

```
Component classification: 0 REUSE, 0 ADAPT, 4 NEW (total 4)

00:00  ✓ layers_0_input_layernorm    1/7
01:34  ✓ layers_0_mlp                2/7
06:16  ✓ layers_0_self_attn          3/7
09:33  ✓ module                      4/7
23:03  ✓ voxtral_codec_decoder       5/7
29:57  ✓ voxtral_flow_matching       6/7
33:48  ✓ voxtral_tts_backbone        7/7

Graduated (ON_DEVICE): 7/7 (100%) actually graduated (native stub, PCC-verified)
RUN ENDED: bring-up complete — gate can_stop
```

**2,553 lines of generated TTNN** across 11 stubs in `models/demos/voxtral_tts_full/`, from a model
the tool had never seen, with `0 REUSE / 0 ADAPT` — nothing was copied; all of it was written.

### The qualification that must travel with that number

**Only 3 of 7 components were gated on REAL inputs.**

```
[capture] selected AutoModelForCausalLM (VoxtralTtsForConditionalGeneration) resolving 4/7
[capture] layers_0_input_layernorm: submodule not resolved; skipping.
[capture] layers_0_mlp:             submodule not resolved; skipping.
[capture] layers_0_self_attn:       submodule not resolved; skipping.
[preflight] captured 3/7 components; per-component PCC tests will use real inputs
```

`voxtral_tts_backbone`, `voxtral_codec_decoder` and `voxtral_flow_matching` got captured IO. The
three decomposed sub-components did not, and graduated against synthetic inputs — the §6.54 trap
(29.5% code flips on synthetic against 3.9% on real), which the tool's own documentation warns about.

### ⚠ F23 CORRECTED — three of the four capture misses were OURS, not the tool's

**Filed first as a tool finding; most of it is not.** The `captured 3/7` result had two causes and
only one belongs to the tool.

**Ours (S5).** The first version of the HF wrapper exposed the backbone's layer stack as

```python
self.layers = nn.ModuleList([nn.Module() for _ in range(26)])   # 26 EMPTY placeholders
```

with a comment saying, in as many words, that this existed *"so a structural walk finds a 26-deep
stack"*. The weights lived in a flat `ParameterDict` and `forward` called the reference function
over it. So the model **advertised structure it did not have**. The tool believed the
advertisement, decomposed `decoder_layer` into `layers.0.input_layernorm` / `.self_attn` / `.mlp`,
tried to hook those paths, and correctly reported:

```
[capture] layers_0_input_layernorm: submodule not resolved; skipping.
[capture] layers_0_mlp:             submodule not resolved; skipping.
[capture] layers_0_self_attn:       submodule not resolved; skipping.
```

Three of the four misses. **The tool's message was precise and correct; the model was lying to it.**

**Fixed.** The backbone is now real per-layer `nn.Module`s whose `forward`s call the reference's own
primitives (`rms_norm`, `split_heads`, `apply_rope`, `gqa_attention`, `merge_heads`, `swiglu`)
composed in `_layer`'s exact order — still bit-exact (prefill, prefill+cache and steps all
`maxdiff 0.0`), now with 138 named modules instead of 27, and verified by *firing hooks on a real
prompt* rather than by assuming:

```
hooks fired: {input_layernorm: 7, self_attn: 7, mlp: 7, flow: 3, codec: 1}
```

— the multiplicities of the actual frame loop, which is the check that would have caught the
placeholder immediately.

**The lesson, and it is general enough to be worth stating in the proposal:** a porting tool reads
the model's declared structure as ground truth. A wrapper that fakes structure to satisfy a
discovery pass will be believed, and the damage surfaces somewhere unrelated — here, as
"synthetic inputs" three stages later. **Verify a wrapper by hooking it, not by listing it.**

**What remains genuinely the tool's**, unchanged, is below.

### F23 — the capture drivers guess, and the config already says they should not

The clearest evidence yet for the representative-inputs recommendation, all from one run:

```
[capture] running drivers with pixel_values shape (1, 3, 224, 224) on 4 hook(s)
[capture] driver `model(pixel_values=...)`: ValueError: give input_ids or inputs_embeds
```

It drove a **text-to-speech** model with a **224x224 image tensor**, against a config declaring
`task: "text-to-speech"` and `modality_in: "text"` (F17 again, third instance).

```
[capture] driver `model(input_ids=..., attention_mask=...) [10 tokens]`:
  AssertionError: prompt has 0 audio placeholders but the preset has 169 rows.
[capture] auto-onboard: closed-loop iteration exhausted after 3 attempts:
  runtime ok but fired 0/3 target(s)
```

The generic driver feeds a 10-token prompt; this model needs one with 169 voice-specific audio
placeholders. The assertion message names the exact command that generates one
(`dump_prompt_ids.py --text '...' --voice <name>`) and **the tool has no way to act on a
remediation printed by the model it is porting**. Three LLM-drafted attempts, none valid.

**One fixture file would have given all seven components real inputs.** The tool already insists on
real activations over random ones — its own docs explain why — and then has no channel through
which a user can supply them.

**Suggested fix (restated, now with evidence):** `--calibration-inputs <path>`. A tensor fixture or
a callable. It is the smallest change on this list with the largest effect on correctness, because
every downstream PCC number inherits the quality of these inputs.

### Honest scope of this result

- **This is the tool plus five of my patches** — F1, F2, F5 (earlier), F18/F19 routing, F21
  `trust_remote_code`. Stock, it refuses this model at `compat` and again at the demo loader.
- The graduation gate is **per-component PCC against captured reference IO**. It is not end-to-end
  audio, and the tool has no WER, no MOS and no exact-match check on discrete codes (O9).
- `emit-e2e` — the independent grader — had not yet reported when this was written.

---

## F21 — `trust_remote_code` is a ONE-MODEL allowlist, and the two halves of the pipeline disagree

**Status: FIXED in this checkout** · severity: any custom-architecture checkpoint passes preflight
and dies in the demo · reported: not yet

```
FAILED models/tt_transformers/demo/simple_text_demo.py::test_demo_text[...]
  ValueError: The repository /localdev/.../voxtral-tts-full contains custom code which
  must be executed to correctly load the model.
```

The bring-up half loads with custom code enabled — `bringup_loop.py:486`,
`_cls.from_pretrained(HF_MODEL_ID, trust_remote_code=True, ...)`. The demo half, via
`model_config.py`, enables it like this:

```python
if self.base_model_name in ["Phi-3-mini-128k-instruct"]:
    self.trust_remote_code_hf = True
```

**Custom-architecture support is an allowlist containing exactly one model.** So a
`trust_remote_code` checkpoint clears Step 0 (*"transformers can load ... [ok]"*), clears static
analysis, and then fails at execution — and the message blames the repository rather than the
loader's configuration.

**Fix applied:** decide from the checkpoint, not the name. `auto_map` in `config.json` **is** the
declaration that a model ships custom modelling code — HF refuses to load such a model without
`trust_remote_code`, so its presence is decisive and needs no allowlist. `TT_TRUST_REMOTE_CODE=0`
restores the old behaviour for a checkpoint that should not be trusted.

### The theme these three share, and it is worth stating once in the proposal

| finding | decided by | available instead |
|---|---|---|
| **F18** | is `model_type` a known **name**? | do the config's fields describe a decoder? |
| **F19** | is there a template with a similar **name/family**? | does the template's `canonical_hf_id` equal the model being ported? |
| **F21** | is the model's **name** on an allowlist? | does `config.json` declare `auto_map`? |

**Three gates, three times deciding by identity when the answer was available from structure.** In
each case the structural signal is present in data the tool has already loaded, and in each case
the name-based answer fails on the first model that is not already known to it — which is the exact
population a porting tool exists to serve.

### F19 (addendum) — the generic demo has a SECOND entrance

The routing gate patched under F19 fired correctly this run:

```
NOT routing to the generic LLM demo: family was INFERRED from config structure ...
```

and the model still reached `simple_text_demo`, via a different path — `scaffold` raising
`ColdStartScaffoldError`, which the CLI handles as *"COLD-START PATH (no per-model `tt/` folder
needed)"* at `cli.py:9189`. **Two independent routes reach the same generic backend, and closing
one does not close the other.** Recorded rather than patched: one gate is a defect, two gates
reaching the same place by different reasoning is a design note the author should see.

---

## F22 — the isolation worktree silently ignores uncommitted changes to the tool's own source

**Status: found 2026-08-14, cost one run** · severity: developer iterating on the tool gets stale
behaviour with no warning · reported: not yet

`auto-up` runs in a private worktree — a good design, and the reason nothing it does can damage the
caller's checkout:

```
[isolation] worktree: /tmp/tt_hw_planner__..._1786737198
```

That worktree is created from **`HEAD`**. An edit sitting in the working tree is not in `HEAD`, so
it does not exist inside the run — **and nothing says so**.

Concretely: the F21 patch was written to `models/tt_transformers/tt/model_config.py`, compile-checked,
and the run launched. It failed with the *identical* `ValueError` the patch fixes. The worktree was
on `81814c5383` while the patch landed in `5ee438f04b`; `grep -c auto_map` in the worktree returned
**0** against **3** in the main checkout.

**Why this matters more than an ordinary footgun.** The people most likely to edit this tool's source
are the people extending it — adding a family, a block, an op-registry entry — and the natural loop
is *edit, re-run, observe*. That loop silently observes the previous version. The failure looks
exactly like "my fix didn't work", which is the most expensive possible misdiagnosis.

**Suggested fixes, cheapest first:**

- At worktree creation, if `git status --porcelain` is non-empty for tracked tool source, print a
  one-line warning naming the files that will NOT be included.
- Offer `--include-uncommitted` (a `git stash`-and-apply, or worktree-from-working-tree).
- Print the worktree's commit sha in the banner. It is already printing the path; the sha is the
  thing that determines behaviour.

**Verification I now use before every relaunch:** grep the created worktree for the patch itself,
rather than trusting the main checkout —

```
worktree: /tmp/tt_hw_planner__..._1786737523
patch present in worktree: 3        # was 0 on the run that failed
```

### Process note (mine)

I launched the run before committing the patch, then committed while it was in flight. Ordinary
sequencing error, but it is worth recording next to the finding: the tool's design made a routine
mistake produce a result indistinguishable from a failed fix, and I only caught it because the error
message was byte-identical to the previous run's — which is a weak signal to rely on.

---

## ★ F25 — decomposition children lose their parent's path prefix, and the plan is copied from another model

**Status: found 2026-08-15, worked around; the real fix is one line** · severity: silently degrades
per-component gates to synthetic inputs · reported: not yet

Two independent defects that compound, both visible in the same log.

### (a) The tool computes the correct path and then discards it

```
line 472  [recompose-link] `decoder_layer` (backbone.layers.0) -> 3 on-device child component(s)
line 468  [reinject] re-added decomposition child `layers_0_input_layernorm`
                     (layers.0.input_layernorm) of `decoder_layer`
line 487  [capture] layers_0_input_layernorm: submodule not resolved; skipping.
```

The recompose-link step records the parent **fully qualified** — `backbone.layers.0`. The reinject
step records the children **relative** — `layers.0.input_layernorm`. The capture hook uses the
children's path, looks up `layers.0.input_layernorm` on a model whose stack lives at
`backbone.layers.*`, finds nothing, and skips.

**The correct path is four lines away in the same log.** Children should inherit the parent's
qualified prefix.

### (b) `decomposition_plan.json` is COPIED from the closest existing demo

```
line 202  A  models/demos/voxtral_tts_full/decomposition_plan.json
line 203        copied from models/demos/voxtral_tts_backbone/decomposition_plan.json
```

and that file contains, correctly for **its own** model:

```
"layers.0.input_layernorm", "layers.0.mlp", "layers.0.self_attn"
```

`voxtral_tts_backbone` is a bare `MistralForCausalLM` whose stack genuinely is at top level. The
three-block model's is not. So a plan describing one model's topology was applied to another's —
**F19's template substitution, resurfacing in the decomposition plan rather than the demo.** The
demo it copied from is itself a previous artifact of this same tool.

### Consequence

Three components could not be hooked, so they graduated against **synthetic** inputs while the run
reported success — the §6.54 trap, reached without anyone making a mistake at the point of failure.

### Fixes

1. **Qualify child paths with the parent's prefix** at reinject time. The value is already computed.
2. **Do not copy `decomposition_plan.json` across models.** Regenerate per model, or at minimum
   validate every recorded path resolves against the model being ported and discard the plan if not.
3. Cheap defence that catches both: **after building the hook list, assert each path resolves**, and
   fail loudly rather than `skipping`. A skipped hook silently changes what the PCC gate measures.

**Workaround used here:** the Block-1 demo was moved out of `models/demos/` so nothing could be
copied from it, forcing the plan to regenerate against the model actually being ported.

### CONFIRMED — both fixes were necessary, and each was necessary alone

Measured across three runs of the same model, changing one thing at a time:

| run | model structure | template pool | capture result |
|---|---|---|---|
| 1 | 26 empty `nn.Module()` placeholders (**ours**, S5) | Block-1 demo present | `resolving 4/7`, **captured 3/7** |
| 2 | real per-layer submodules | Block-1 demo present | `resolving 7/10`, still 3 × `submodule not resolved` |
| 3 | real per-layer submodules | **empty** | `copied from (skeleton — no sibling source)` · **`resolving 7/7`**, zero unresolved |

**Neither fix alone was sufficient.** Real submodules did not help while the plan carried another
model's paths; deleting the stale plan would not have helped while the modules were hollow. That is
worth stating to the PR author, because it is why this failure is hard to diagnose from a single
run: two independent causes produce one identical symptom (`submodule not resolved`), and fixing
either one leaves the symptom unchanged.

**A side benefit worth noting:** with real modules to inspect, classification moved from
`0 REUSE, 0 ADAPT, 4 NEW (total 4)` to **`3 REUSE, 0 ADAPT, 4 NEW (total 7)`** — the tool recognised
three components as things `tt_transformers` already implements rather than writing them from
scratch. Honest structure did not just fix the gate; it made the port cheaper.

### Run 4 — `captured 7/7`, and the third fix was also ours

Runs 1-3 reached 3/7, then 5/7. The last two misses were **Blocks 2 and 3**, and the cause was again
the wrapper, not the tool:

```
driver `model(input_ids=..., attention_mask=...) [10 tokens]`:
   AssertionError: prompt has 0 audio placeholders but the preset has 169 rows
driver `submodule[backbone](**['inputs_embeds'])`: ok          <- backbone ONLY
```

Unable to construct a valid whole-model prompt, the framework fell back to driving the **backbone
submodule alone** — which reaches every backbone component and never executes the flow matcher or
the codec, so those two were gated on synthetic inputs while the run reported success.

**The model could not be run with no arguments.** Its `forward` required a prompt whose
audio-placeholder count is voice-specific (169 rows for the default voice), and no such prompt was
shipped with it. A generic driver cannot invent one. Fixed by carrying `default_prompt_ids` in
`config.json` — deliberately in the config rather than a sidecar file, because trust_remote_code
copies only `.py` into its module cache (S4 #4), so anything resolved from `__file__` is absent.

```
[capture] tts_backbone / decoder_layer / r_m_s_norm / attention / m_l_p /
          codec_decoder / flow_matching:  captured
[preflight] captured 7/7 components; per-component PCC tests will use real inputs
```

**Trajectory: 3/7 → 5/7 → 7/7, across three independent causes** — two ours (hollow modules, no
default prompt), one the tool's (F25's copied decomposition plan). Every per-component gate is now
measured against tensors the deployment actually produces.

**A caveat that keeps F23 intact.** The driver that finally succeeded is
`model(pixel_values=...): ok` — the *image-tensor* driver, on a text-to-speech model. It works only
because `forward` ignores unknown kwargs and falls through to the bundled default. **The tool still
does not know how to drive this model; it merely can no longer fail.** The recommendation is
unchanged: a `--calibration-inputs` channel, so representative inputs are supplied rather than
guessed.

**Packaging lesson for anyone wrapping a model for this tool:** it must be runnable with **no
arguments**. Every automatic driver, capture and smoke test depends on that, and a model whose
`forward` demands a specially-constructed input is undrivable no matter how correct it is.

### ⚠ CORRECTION — `captured 7/7` does NOT mean the tests use the captured tensors

I reported that reaching `captured 7/7` put every per-component gate on deployment tensors. **It does
not.** The capture succeeded — 23 files on disk under `_captured/` — and **none of the seven tests
read them**:

```
test_attention       captured-refs: 0   synth-refs: 5
test_codec_decoder   captured-refs: 0   synth-refs: 5
test_flow_matching   captured-refs: 0   synth-refs: 5
    ... all seven identical
```

`captured N/M` counts **recordings made**, not recordings consumed. Two different things, and I
conflated them.

**There are three tiers of input quality here, not two:**

| tier | what it is | where the run actually is |
|---|---|---|
| 1 | random tensors from name-guessing (`_make_arg_for`) | where it started — tests **crashed**: `cis` got `randn(1,64,3072)` where a COMPLEX rope table was required |
| 2 | inputs built from the reference's OWN primitives (`rope_cis`, `causal_bias`) | **where all 7 component tests are** |
| 3 | the recorded deployment activations | captured, on disk, **unused** |

**And tier 3 is declined for a genuine reason**, which the agent-rewritten harness documents:

> *`_captured/attention/args.pt` holds a real deployment step: `h=[1,1,3072]` with a 208-deep KV
> cache. It is not usable as-is for a unit test. The cache dict is **MUTATED** by
> `VoxtralAttention.forward`, and the harness hands the same object to the torch reference and then
> to the ttnn stub*

Feeding one mutable cache to the reference and then to the stub means the stub sees the reference's
write — a comparison against contaminated state that would look correct. Declining the capture is
the right call; **silently substituting tier 2 while the run reports `captured 7/7` is not.**

**F26 — report what the gate MEASURED, not what was collected.** A line reading `captured 7/7`
directly above per-component PCC results invites exactly the reading I gave it. The gate should
state its input provenance per component — `real-capture` / `synthetic-from-reference` /
`synthetic-guessed` — because those three carry very different confidence and §6.54 measured the
difference at 29.5% vs 3.9% error on the same code.

**What is unaffected:** the END-TO-END test genuinely uses the real prompt through the whole
pipeline and compares waveform against waveform. That is the number that decides whether the audio
is right, and it is honest.

---

## ★ F27 — the captured input is DISCARDED where one `deepcopy` would have kept it

The harness captures a real deployment activation for `attention`, correctly works out that it
cannot hand the same object to both sides, and then **throws it away** rather than copying it.

Its own note, in full:

> *`_captured/attention/args.pt` holds a real deployment step: `h=[1,1,3072]` with a 208-deep KV
> cache. It is not usable as-is for a unit test. The cache dict is MUTATED by
> `VoxtralAttention.forward` (`cache[cache_key] = (k, v)`), and the harness hands the same object to
> the torch reference and then to the ttnn stub — so the stub would attend over a cache one position
> longer than the golden did. **Dropping the cache instead makes the test vacuous**: at S=1 with no
> cache the softmax is over a single key, so it returns 1.0 whatever q and k are, and RoPE — the
> thing most likely to be wrong in a port — stops affecting the output at all.*

It considers exactly two options — **share the object** (contaminated) or **drop the cache**
(vacuous) — and takes neither, substituting a synthetic 64-token causal prefill.

**The third option is absent.** Give each side its own copy:

```python
ref_out  = reference(h, cis, bias, deepcopy(cache), key)
stub_out = stub(h, cis, bias, deepcopy(cache), key)
```

`grep` confirms it never occurred to the harness: **no `deepcopy`, no `.clone()`, no `copy.` anywhere
in `tests/pcc/conftest.py`.** The cost is negligible — one layer's cache at 208 positions is
≈ 208 × 8 heads × 128 dims × 2 tensors × 4 B ≈ **1.7 MB**.

**And the copy would be strictly better than the substitute**, in exactly the dimension that
matters. The synthetic prefill exercises RoPE at positions 0-63 with an empty cache; the real
captured step exercises it at **position 208 with a 208-deep cache**. RoPE errors are
position-dependent — `[gpt-21]` records SDPA settings that were correct at one length and
*"NOT SAFE — position sweep"* across others. The harness itself calls RoPE *"the thing most likely
to be wrong in a port"*, and then tests it at the positions least likely to expose the bug.

**Fix:** deep-copy mutable captured args per side. One line, and it converts the whole capture
pipeline from collected-but-unused (F26) into actually-used.

*(Credit to the reviewer who spotted this: the harness's reasoning is sound about why sharing fails
and why dropping fails, which makes the missing third option easy to overlook.)*

## ★★ F28 — the entire end-to-end verdict rests on ONE prompt

```
tests/e2e/test_e2e_pipeline.py     pytest parametrize: 0
CLI flags for prompts / cases:     none found in scripts/tt_hw_planner/cli.py
```

One text, one voice, one seed, one horizon. For a **generative speech model**, that is the whole
correctness gate — `Verdict: PASS` is decided on a single utterance.

**For contrast, the hand-port's gate on the same model** runs **45 utterances across 3 seeds**, and
its own history says why: `§6.21` records that a case's frame count depends on what ran before it in
the same process, so arms need identical history; `§6.62`'s `tail_probe.py` exists specifically to
*"count failures, not means"* because damage concentrates in rare bad utterances; and the
`w2 -> bf8_b` experiment moved `mos_min` by **0.245** while `mos_longform` moved 0.021 — a
mean-preserving change that mauled the worst case. **A one-utterance gate cannot see any of that.**

This compounds every other correctness finding in this document:

- **O9** — no WER, no MOS, no exact-match on discrete codes. Now also: no sample size.
- **F26** — per-component gates run on synthetic inputs, so the e2e test carries the correctness
  burden alone. It carries it on n=1.
- **F27** — the one component whose real input WAS captured has it discarded, so even that single
  sample is not deployment-representative at the component level.

**Suggested fix, in order of cost:**

1. `pytest.mark.parametrize` the e2e test over a prompt list, and take the **worst** PCC as the
   verdict rather than the only one.
2. A `--eval-prompts <file>` flag, so a user supplies the set — the same channel F23 asks for on the
   input side.
3. Report the distribution (min / mean / n), not a scalar. A single number invites exactly the
   confidence it cannot support.

**Why this is arguably the most important finding here:** every other defect in this document is a
thing the tool does wrong that a reader could catch. This one is a thing it *doesn't do*, and its
absence is invisible — the report says `PASS` and shows a PCC, and nothing on the page hints that
`n=1`.

### F28b — PROPOSAL: get the sample size from the BATCH dimension, not from N sequential runs

The obvious objection to F28 is cost: running 45 utterances the way the hand-port's gate does takes
~18 minutes, and an inner-loop correctness gate cannot afford that. **The batch dimension makes it
close to free**, and for this model two of the three blocks already support it:

```
Block 2  predict_velocity   [B,36], [B,3072], [B,3072] -> [B,36]
         semantic_code      h [B,3072] -> [B,1]
         decode_frame       [B,1], [B,3072] -> [B,36]              <- already batched
Block 3  reference_decode   codes [B,37,T] -> waveform [B,1,T*240*8]  <- already batched
Block 1  reference_forward  [1, S, 3072] -> [1, S, 3072]           <- pinned at 1
```

Block 2 is *already* running batched in production — CFG folds the batch to 2x (§6.35). So the gate
would be exercising a path the model already uses.

**What blocks Block 1 is deployment concerns a TEST does not have:**

| deployment problem | why a gate can ignore it |
|---|---|
| prompts differ in length | pad to a common length; the causal mask already handles padding |
| each utterance stops at its own `[END_AUDIO]` | run a FIXED frame count and ignore termination |
| per-sequence retirement scheduling | not needed when every row runs the same number of frames |

And the shape works out: a tile is 32 rows, so **B <= 32 still occupies one tile**. `per_core_M=1` /
`fuse_batch=True` are not violated. `nlp_create_qkv_heads_decode` already emits
`[1, batch, heads, head_dim]`, and `paged_update_cache` / `sdpa_decode` both take a batch dimension.
Batch-5 is untested here, not structurally blocked.

**The proposal:**

1. Run the e2e gate at **B = 5-8 prompts**, padded, fixed horizon, reference batched identically.
2. Report **min / mean / n** across rows, and gate on the **worst** row, not the mean — §6.62's
   `tail_probe.py` exists because damage concentrates in rare utterances, and `w2 -> bf8_b` moved
   `mos_min` by 0.245 while `mos_longform` moved 0.021.
3. Cost is roughly one utterance's wall time for 5-8 samples, which is what makes it viable as an
   inner-loop gate rather than a nightly one.

**Two caveats the PR author should hear with it:**

- **A port validated only at B=1 may silently assume it.** If the generated stubs break at B=5, that
  is itself the finding — and a cheap one to surface, since the gate would catch it on day one.
- **It changes the performance regime.** Batching is a throughput lever, so a B=5 measurement is not
  comparable to B=1 deployment timing. Use it for CORRECTNESS only; letting `optimize` tune against
  a batched measurement would optimise the wrong operating point.

*(Proposed, deliberately NOT implemented here — this is a design suggestion for the PR, not a change
we validated.)*

---

## ★★★ F29 — the threshold does not just GATE quality, it SETS it. And the two defaults disagree.

The single cleanest experiment in this document, and the most actionable finding for the PR.

### The two defaults

```
cli.py:10986      pe2e.add_argument("--pcc-target", type=float, default=0.95,
                     help="PCC threshold for the final HF-vs-TT comparison (default: 0.95)")

e2e_mcp.py:20     E2E_MCP_PCC   required e2e PCC threshold (default 0.99)
e2e_mcp.py:43     _PCC = float(os.environ.get("E2E_MCP_PCC", "0.99"))
```

The gate engine documents **0.99** as *"required"*. The CLI passes **0.95** and overrides it. A user
who never touches the flag gets the loose one, and nothing says a stricter default exists.

### What that costs, measured

Same port, same model, same machine. Only the threshold changed:

| threshold | measured e2e PCC | rounds the fix-loop needed |
|---|---|---|
| `0.95` (CLI default) | **0.9586** | `rounds=1 can_stop=True` — passed immediately, loop never worked |
| `0.99` (engine default) | **0.9986** | round 1, 45+ tool calls of actual repair |

**The 0.9586 was not a ceiling.** It was not a precision limit either — the test's own
device-precision bound reads `1.0000 at N=4`, and the comparison ran at N=4 (waveform 7680 samples
at `SAMPLES_PER_FRAME=1920`). It was simply **where the loop stopped, because the gate let it.**

Given a target it could not trivially clear, the same tool on the same code found another four
points of accuracy.

### Why this is the important version of O8

O8 recorded that the accept test has no exchange rate — it keeps any change that is faster and above
the floor. This is the mirror image and it is worse: **the threshold is not a floor, it is the
target.** Quality delivered ≈ quality demanded. A default set four points below the engine's own
documented requirement therefore does not merely permit worse ports — it *produces* them.

For a model where `§6.31` records that one flipped semantic code redirects an entire utterance, and
where the same port at 0.95 shipped `code exact-match: all codebooks 0.8649` — one acoustic code in
seven differing, against the hand-port's measured `codes_real_pct 5.2` — that gap is not academic.

### Fixes

1. **Make the CLI inherit the engine's default rather than override it.** One line. If 0.99 is
   documented as *required*, the CLI should not silently ask for less.
2. **Print the threshold's provenance in the report** — `pcc>=0.95 (CLI default; engine default is
   0.99)`. The banner currently states the number with no indication that it was lowered.
3. **Derive the floor from the measured precision bound.** The test already computes it
   (`1.0000 at N=4, 0.9458 at the 8-frame cap`). A fixed constant is either unreachable or too
   generous depending on the horizon; the bound is neither.

*(Recommendation 3 matters because 0.99 is NOT universally safe: at the 8-frame horizon this model's
own reference is only reproducible to 0.9458, so a hard 0.99 would be unsatisfiable there. The right
target is a function of the measured bound, not a constant.)*

---

## ★★★ F30 — the drift gate exists, detects the stale template, and is wired never to block

**Status: live in this checkout** · severity: bring-up selects a template directory that does not
exist, and says so only in a line it also suppresses · reported: not yet

The tool ships a registry drift check whose entire stated purpose is to stop this. Run against this
tree on 2026-08-15 it works perfectly:

```
$ python -m scripts.tt_hw_planner sync-registry --check
  [MISSING] family_backends[Voxtral TTS Backbone (mistral decoder)].demo_path
            -> models/demos/voxtral_tts_backbone
[sync-registry] FAIL: 27 registry path(s) missing from the checkout — fix the registry or restore the paths.
rc=1
```

`sync_registry.py:1-8` says why it exists: *"``--check`` exits non-zero on hard drift (a mapped path
that is gone) so CI / a pre-plan gate fails loudly instead of the planner silently mis-pointing at a
stale sibling."*

`up` / `auto-up` reach the same function through `_warn_on_registry_drift()` (`cli.py:8103`), whose
docstring states the opposite as a design commitment:

> *"Never raises: neither a fetch nor a drift check may block bring-up."*

On hard drift it prints exactly one line (`cli.py:8142-8149`):

```
[registry] N mapped registry path(s) are stale on this checkout — run `tt_hw_planner sync-registry` for detail.
```

…followed by the full `format_drift(issues)` listing, naming every stale path. **Verified against a
live `auto-up` on 2026-08-15**, which printed all 26 before proceeding.

That the detail appears at all is an accident worth its own line. It is guarded by
`if os.environ.get("TT_HW_PLANNER_VERBOSE")` (`cli.py:8147`), and the default is set at
`cli.py:8082` as:

```python
os.environ.setdefault("TT_HW_PLANNER_VERBOSE", "0")
```

The string `"0"` is **truthy** in Python, so the guard is always true and verbose output is
permanently on for this branch — including for a user who sets `TT_HW_PLANNER_VERBOSE=0` explicitly
to turn it off. The check wants `not in ("", "0", "false")`, the idiom the same file already uses
for `TT_HW_PLANNER_NO_WRAP` (`__main__.py:34`).

**So the tool prints everything it knows and proceeds anyway** — which is the finding in its
strongest form. This is not a reporting gap that hides the problem; the operator is shown 26 stale
paths, by name, and the run continues into template selection regardless. The whole body is also
wrapped in `except Exception: pass`, so a drift check that itself throws is indistinguishable from
a clean checkout.

### What it cost here, measured

`models/demos/voxtral_tts_backbone/` was removed from this tree at 09:29 on 2026-08-15. The bring-up
run at 15:11 selected it anyway — `models/demos/voxtral_tts_full/RUN_REPORT.md`:

```
Backend picked:    Voxtral TTS Backbone (mistral decoder)  (TEMPLATE-FALLBACK — model_type mismatch)
Closest template:  models/demos/voxtral_tts_backbone/        <- absent from the checkout
Sibling base:      /localdev/lserbedzija/hf_models/voxtral-tts-backbone (model_type=mistral)
```

This is **F19 with the safety net already built and switched off.** F19 showed template dispatch can
silently run a different model; F30 shows the tool can *prove* the template is gone, on the same
run, and proceed regardless. It is also F20's exact shape a third time: the knowledge exists and is
wired to stdout instead of to control flow.

**This is not only our mess.** The dangling Voxtral entry was ours (S7) and we removed it. The
drift check still fails afterwards, with **26 mapped paths missing** — `XTTS-v2 (multilingual TTS)`
→ `models/demos/xtts_v2`, `tt_dit/minimax_h3 (auto-upstream)`, and 24 more, all entering the
registry through the tool's own commits (`589a4d121a`, `12bd4e4ef8`). So the shipped registry
points at 26 paths that do not exist in the checkout it ships with, and the gate that knows this is
the one guaranteed never to fire. Any of those 26 can be selected as a template exactly the way
ours was.

### Fixes

1. **Make hard drift on the *selected* backend fatal.** Global drift can stay advisory — 27 stale
   paths in unrelated families should not block a bring-up. But once template selection has
   *picked* an entry, a missing `demo_path` on that entry is not a warning, it is a broken run.
2. **Fix the verbosity guard** — `os.environ.setdefault("TT_HW_PLANNER_VERBOSE", "0")` plus a bare
   truthiness test means the flag can never be off, and `TT_HW_PLANNER_VERBOSE=0` does not turn it
   off. Compare against `("", "0", "false")` as `__main__.py:34` already does. (The drift detail
   itself should stay visible — print it unconditionally rather than by accident.)
3. **Narrow the `except Exception: pass`.** A drift check that crashes currently reports as a clean
   checkout.

---

## ★★ F31 — the profiler reports a missing CSV where the child actually died of a bus error

**Status: live** · severity: the optimizer agent is handed a plumbing error instead of a crash ·
reported: not yet

`termination_check()` returned this to the optimizer agent at 16:27:45 on 2026-08-15:

```
can_stop: false
error: "profiler crashed: tracy run exit 1 (log: /tmp/perf_mcp_4tvrspfx/run0_tracy.log)
        AssertionError: cpp_device_perf_report.csv not found and legacy device log
        profile_log_device.csv is also missing in /tmp/perf_mcp_4tvrspfx/tracy_out/.logs."
```

Read literally that is a profiler-output-plumbing problem, and an agent told to keep optimizing will
go looking for one. It is not what happened. The surviving log of a sibling run
(`/tmp/perf_mcp_h4kudb_0/run0_tracy.log`) shows the profiled child aborting mid-forward:

```
Fatal Python error: Bus error

Current thread (most recent call first):
  ttnn/ttnn/decorators.py:650 in __call__
  models/demos/voxtral_tts_full/_stubs/m_l_p.py:38 in __call__
  models/demos/voxtral_tts_full/tt/pipeline.py:165 → 195 → 369 decode_stack → 413 run_tts
  tests/e2e/test_main_perf.py:204 in _eager_forward → 251 in test_main_perf
Aborted (core dumped)
```

The CSV never appears **because the process died before writing it.** The postprocess then walks its
fallback chain — `process_ops_logs.py:1136` warns that `cpp_device_perf_report.csv` is missing and
falls back to legacy parsing, `process_ops_logs.py:755` finds that missing too and raises — and the
raise is the only thing that reaches the caller. The abort, the signal, and the stack are all in the
log the caller cites but does not read.

**The masking is the finding, not the bus error.** Whether the bus error is a ttnn defect, a
profiler-buffer overrun on a three-block forward, or a fault in our own hand-written perf test is
not established here and is not claimed. What is established is that an abort was reported as a
missing file.

### Fixes

1. **Check the child's exit status first.** A child that exited by signal (or non-zero) should be
   reported as that — `tracy child aborted (SIGBUS) at <last stack frame>` — before any assertion
   about its outputs runs.
2. **Include the log tail in the error.** The caller is already given the log path; the last ~40
   lines would have carried `Fatal Python error: Bus error` into the agent's context.
3. **Do not let the fallback chain's terminal assertion be the reported cause** when an earlier,
   more specific failure was already observed.

---

## ★★ F32 — `termination_check()` blocks for 30 minutes with no progress channel, and the retry never returns

**Status: live** · severity: an unattended optimizer run cannot be distinguished from a hung one ·
reported: not yet

Timings from the driving session's own transcript (`0b038219-…jsonl`), 2026-08-15:

| time | event |
|---|---|
| 15:57:53 | agent calls `termination_check()` |
| 16:27:45 | returns — **29 min 52 s later** — with the F31 error |
| 16:28:02 | agent retries `termination_check()` |
| — | never returns; the transcript ends here |

Nothing is emitted in between. The tool re-profiles the model inside the call, so half an hour of
silence is the *normal* case, not the failure case — which means the failure case is
indistinguishable from it. The run above was abandoned by its operator as hung; it had in fact
returned one error and was sitting inside a second identical call.

This is **F7 recurring at the optimizer stage** (*"all progress flows through one channel, and
nothing notices when it is dead"*), and it compounds F31: the one message that does come back after
thirty minutes describes the wrong failure.

### Fixes

1. **Emit progress from inside the call** — at minimum the sub-step (`profiling`, `parsing`,
   `checklist`) and the elapsed time.
2. **Bound the call and return partial status** rather than blocking indefinitely on the retry.
3. **Make a repeated identical call cheap or refused.** The retry re-ran the same 30-minute profile
   against an unchanged tree.

---

## ★ F33 — `worktree-list` can never print ORPHAN, so dead worktrees accumulate looking healthy

**Status: live in this checkout** · severity: the operator is told there is nothing to reclaim while
2.7 GB is reclaimable · reported: not yet

Six bring-up worktrees on this box, one per `auto-up` run:

```
$ python -m scripts.tt_hw_planner worktree-list
  /tmp/tt_hw_planner__…_1786735901   …voxtral-tts-full   1541336   22.0   active
  /tmp/tt_hw_planner__…_1786736367   …voxtral-tts-full   1548859   21.9   active
  … 4 more, all "active"

$ for p in 1541336 1548859 1552915 1557766 1797227 1801185; do ps -p $p ...; done
  1541336 DEAD   1548859 DEAD   1552915 DEAD
  1557766 DEAD   1797227 DEAD   1801185 DEAD
```

Every creator is dead. Every row says `active`. The cause is one expression —
`commands/worktree_list.py:20`:

```python
status = "ORPHAN" if id(s) in orphans else "active"
```

`id(s)` is CPython's builtin object-address `id()`. `list_orphans()` (`worktree.py:169-174`) returns
`List[WorktreeSession]` — objects, not addresses. An `int` is never `in` a list of
`WorktreeSession`, so the ORPHAN branch is unreachable and every worktree prints `active` forever.

**This is F2's shape again**: a verdict that cannot fire, where the failing branch is the one that
signals work is needed.

**The two commands contradict each other in the same checkout.** Immediately after `worktree-list`
called all six `active`, `worktree-cleanup` was run and printed, for the very same PIDs:

```
orphan worktree: /tmp/tt_hw_planner__…_1786735901 (… creator-pid=1541336 dead, age=22.1h)  -> removing
… removed 6 orphan worktree(s)
```

Same predicate, same process, opposite answers — because cleanup asks `list_orphans()` and the
listing asks `id()`. 2.7 GB was reclaimable the whole time the listing said otherwise.

**The reclaim path itself is correct.** `cleanup_orphans()` (`worktree.py:195`) calls
`list_orphans()` and iterates the objects properly, so `worktree-cleanup` *does* remove them. The
damage is confined to the display — but the display is the only thing telling an operator whether
running cleanup is worthwhile, and it says no. Six worktrees × ~430 MB accrued unnoticed, one per
run, on a box where the model checkpoint alone is 16 GB.

**Secondary, and worse in a shared setting.** `_pid_alive()` (`worktree.py:176-186`) treats
`PermissionError` as not-alive:

```python
    except (ProcessLookupError, PermissionError):
        return False
```

`os.kill(pid, 0)` raises `PermissionError` precisely when the process **exists but belongs to
another user**. That PID is alive. Classified orphan, it becomes a `git worktree remove --force`
(and an `shutil.rmtree` fallback) against a worktree whose creator is still running.

### Fixes

1. **Compare identity, not `id()`** — `if s in orphans`, or match on `s.path`.
2. **`PermissionError` means alive.** Only `ProcessLookupError` means gone.
3. **Have `worktree-cleanup` print the same status `worktree-list` computes**, so the two can never
   disagree about what is reclaimable.

---

## ★★★ F34 — deleting the model does not delete the model: the overlay store silently restores it, and a from-scratch run is unreachable

**Status: live in this checkout** · severity: two runs from the same HEAD start from different
states, and the difference is invisible · reported: not yet

To re-run the pipeline cleanly at PCC 0.99, `models/demos/voxtral_tts_full/` was deleted and the
deletion **committed** (`42e9bee5f7`). HEAD contained no demo directory; `git status` was clean.
`auto-up` was then launched.

The isolation worktree was created from that HEAD — correctly, and `git log` inside it confirms
`42e9bee5f7`. Then one line went past:

```
  [isolation] applied 0 _shared + 1 model overlay(s)
```

After that line, the worktree contained the **entire previous port**: 63 files, including every
graduated stub, their `.best_native` / `.last_good_native` graduation snapshots,
`.bringup_cc_state.json` (16234 B), `bringup_status.json` (5928 B) and the previous run's
`RUN_REPORT.md` — all stamped with the current run's timestamp, all reinstated on top of a HEAD
that does not contain them.

The overlay store had retained a whole-directory patch for the model. Nothing in the run says a
previously-ported demo directory has just been reinstated; the notice is a count of overlays.

**Why this is worse than a surprising default**

1. **A from-scratch run is not reachable through the documented surface.** Delete the model's
   directory, commit, re-run — and the model comes back. There is no `--no-overlays` /
   `--from-scratch`.
2. **Reproducibility inverts.** Two runs from the same commit produce different starting states
   depending on overlay state that is not in the tree, not in the log, and not in the report.
3. **It is invisible in exactly the place it matters.** The RUN_REPORT records placements and PCC
   for what it *believes* it built this run.

`--reverify` does mitigate part of it — it clears restored graduation snapshots so each component
re-earns its gate — and we passed it. But it is opt-in, it addresses only the markers, and the
restored *implementations* remain regardless. A run that begins with a finished port is not a
bring-up, whatever the markers say.

**The documented wipe does not wipe.** `overlay-drop <model_id>` is documented as *"Omit rel_path to
wipe ALL overlays for the scope."* Run against this model it dropped every patch and left:

```
scripts/tt_hw_planner/overlays/_localdev_…_voxtral-tts-full/locked_modules.json
  {"decoder_layer": {"locked_ts": 1786786489.8,
                     "reason": "children all on device; recomposed as whole-module target"}}
```

A pin from the previous run, recording a structural decision about `decoder_layer`, surviving the
command whose stated job is to wipe the scope. It had to be removed by hand.

**A tell in the same log.** The successful overlay was preceded by ~30 lines of the form
`skipped <path> — git apply --check returned rc=1 … already exists in working directory`. The
overlay system was largely failing to apply against a tree it had itself just populated; the patch
that *did* apply was the whole-directory one. The mechanism is noisy about its failures and silent
about its one consequential success.

### Fixes

1. **Say what was restored, not how many.** `restored models/demos/voxtral_tts_full/ (63 files,
   incl. 5 graduation markers) from the overlay store` is one line and ends the entire class of
   confusion.
2. **Provide `--no-overlays` (or `--from-scratch`)**, and name it in the bring-up docs as the way to
   reproduce a clean port.
3. **Make `overlay-drop <scope>` empty the scope** — `locked_modules.json` included — or state what
   it deliberately preserves and why.
4. **Never carry graduation markers in an overlay.** Ported source is legitimate to reuse; a
   "this component already passed its PCC gate" marker earned in a different run under a different
   threshold is not — see F26 (report what the gate measured) and F29 (the threshold sets quality).

---

## ★★ F35 — backend selection is not reproducible: identical runs pick different templates

**Status: live in this checkout** · severity: the template that shapes the generated port is chosen
non-deterministically · reported: not yet

Two `auto-up` runs, same model, same commit (`42e9bee5f7`), ~4 minutes apart. The deterministic
ranking was **identical** both times:

```
  Sibling candidates (top 2, exact first):
    1. hf_eager universal (TTS)          [score=40; category 'TTS' default (generic runner)]
    2. XTTS-v2 (multilingual TTS)        [score=30; category 'TTS' default]
```

The selection was not:

```
run 1:  Backend match: LLM-RESOLVED  (hf_eager universal (TTS))     <- rank 1
run 2:  Backend match: LLM-RESOLVED  (XTTS-v2 (multilingual TTS))   <- rank 2, score 30
```

Same inputs, same scores, different answer — and in run 2 the LLM ranker overrode its own
deterministic top pick in favour of a candidate scored 10 points lower, with no stated reason
beyond the boilerplate *"the registry-constrained LLM ranker chose this as the closest
architectural sibling"*.

**Why it matters.** The backend is the template, and the template shapes the scaffold: which demo is
copied, which attention/RoPE conventions are assumed, which reuse map applies. F19 already showed
template dispatch can silently run a *different model*; F35 says which template you get is not
stable across identical invocations. A bring-up that cannot be reproduced cannot be bisected, and a
regression cannot be attributed to a code change rather than to the ranker's mood.

**Both winners point at directories that do not exist** — `models/demos/hf_eager/demo.py` and
`models/demos/xtts_v2` are both on F30's list of 26 stale registry paths. The ranker is choosing
between two broken entries and the run continues either way.

**What saved this run.** F18's corrected routing gate declined the generic route in *both* runs,
identically and for the right reason:

```
NOT routing to the generic LLM demo: family was INFERRED from config structure, not a known
model_type; config declares 3 stacks (backbone, flow, codec) and the block checklist covers the
decoder stack only. Scaffolding this model's own stubs instead.
```

So for *this* model the divergence is contained — the port is scaffolded from the model's own
structure rather than from either template. That containment is luck of architecture, not design:
a single-stack model with a known `model_type` would have been routed to whichever template the
ranker happened to name that day.

### Fixes

1. **Make the deterministic score authoritative unless the LLM gives a stated, logged reason to
   depart from it** — and log the reason next to the score it overrode.
2. **Seed / cache the ranker per (model, commit)** so a re-run reproduces the earlier choice, and
   record the resolved backend in the run report as an input, not a narration line.
3. **Exclude candidates whose `demo_path` is missing** before ranking. Both candidates here were
   unusable; the ranker was choosing between two dead links (F30).

---

## ★★★ F36 — "PCC tests will use real inputs" is false: the gate runs on `torch.randn`, and the real capture is never loaded

**Status: live in this checkout, observed during the 0.99 re-run** · severity: every component
graduates against synthetic data while its real activations sit unused on disk · reported: not yet

This is F26 confirmed by direct inspection, and worse than F26 recorded it, because the tool now
*states* the opposite in its own log.

### The claim

Preflight, run 2, 2026-08-15:

```
  [capture] attention: captured args=5 kwargs=0 output=tensor
  …
  [preflight] captured 7/7 components; per-component PCC tests will use real inputs
```

It captured well. `_captured/attention/` and `_captured/decoder_layer/` are **43 MB each** — real
deployment activations with a deep KV cache — and every component directory holds `args.pt`,
`kwargs.pt`, **`output.pt`** (the real reference output) and `manifest.json`.

### What the gate actually runs

`bringup_mcp.py:337` runs the per-component gate as
`_run_focused_pytest(test_files=[tests/pcc/test_<comp>.py])`. That generated test:

- contains **no `torch.load`, no `args.pt`, no `.pt` reference of any kind** — verified across the
  whole `tests/pcc/` directory, 0 hits in 7 files;
- builds every input from the forward signature by **argument name**, via `_make_arg_for`:

```python
if arg_name in ("hidden_states", "inputs_embeds", "embeddings"):
    shape, _ = _detect_hidden_shape(torch_module, model=model)
    return torch.randn(*shape).to(md)
…
if primary is None:
    primary = ("(synthetic)", torch.randn(1, 64, 64))
```

- uses the capture for exactly one thing — deciding **which submodule** to test:

```python
_captured_path = _captured_submodule_path(COMPONENT_NAME)
if _captured_path:
    torch_module = _resolve(model, _captured_path)
```

All six `_captured` references in each test are that path lookup. The tensors are never opened.

### So what graduation at 0.99 means here

A component is graduated when this test reports PCC ≥ 0.99 — against `torch.randn` at whatever
shape the name heuristic infers, compared to the HF module fed *the same* synthetic tensor. For
`attention` that replaces a real 208-deep KV cache with `torch.randn(1, 64, …)`; the captured
`output.pt` that would have made it a true golden comparison is never read.

Raising the threshold does not help. **0.99 on synthetic input is not a stronger claim than 0.95 on
synthetic input — it is a more precise measurement of the wrong thing.** This is the limit of what
F29's threshold fix can buy, and the reason F28's point stands: the whole real-correctness signal
rests on the e2e stage.

### It is not that the captures are unusable

The demo path does consume them — `demo_wiring.py:80-81` requires `args.pt`/`kwargs.pt` to exist,
and `bringup_loop.py` emits a `_load_captured()` helper into generated demo code whose docstring
says it "matches the PCC test convention so the demo passes whenever the PCC test passes". The
convention it claims to match is the one the PCC test does not implement. The plumbing to do this
right is already written and already paid for; the gate simply does not call it.

### Fixes

1. **Load `args.pt`/`kwargs.pt` in the generated PCC test when the capture exists**, and fall back
   to `_make_arg_for` only when it does not — the fallback is the current behaviour, so this is
   additive.
2. **Compare against the captured `output.pt`**, not only against a re-run of the HF module on
   synthetic input. That turns the gate into a golden test at no extra cost.
3. **Make the log line honest.** If the test ran on synthetic inputs, say `captured 7/7; gate ran on
   SYNTHETIC inputs (captures used for submodule resolution only)`. Reporting what the gate
   measured rather than what was collected is F26, and this is the same sentence needing the same
   repair.
4. **State it in the report too.** `RUN_REPORT.md` records components as "graduated, native ttnn,
   PCC verified" with no indication of what they were verified against.

---

## S6 — OURS: our own `conftest.py` bootstrap shadows the built `ttnn` inside the planner's scratch copy

**Status: FIXED (uncommitted at time of writing)** · not a tool defect

`_bootstrap_ttnn_import_paths()` in the repo-root `conftest.py` is ours — added 2026-08-15 in
`bb71494984`, absent from `origin/main` and from PR #46283. It published `<this tree>/ttnn` onto
`sys.path` whenever `ttnn/ttnn/__init__.py` existed.

The hw-planner copies the checkout into a scratch root (`/tmp/tt_hw_planner__<model>_<ts>/`) and runs
pytest from there. That copy carries the `ttnn` **sources** but no compiled `_ttnn*.so`, so the
bootstrap put a source-only regular package ahead of the real one and every test in the copy died at
`ModuleNotFoundError: No module named 'ttnn._ttnn'`.

Fixed by selecting the `sys.path` entries from whichever tree actually holds `_ttnn*.so` (falling
back to `TT_METAL_HOME`) while keeping the calling tree first, so `models`/`tests` still resolve to
the copy under test.

**Recorded because it is easy to mistake for a tool defect** — the failure only appears inside the
planner's own scratch copy, and the tool is what creates that copy. The bug is ours.

---

## S7 — OURS: parking the Block-1 demo left its registry entry dangling

**Status: open at time of writing** · not a tool defect — but it is what exposed F30

`family_backends.py:289-299` registers `Voxtral TTS Backbone (mistral decoder)` with
`demo_path='models/demos/voxtral_tts_backbone/'` and
`canonical_hf_id='/localdev/lserbedzija/hf_models/voxtral-tts-backbone'`. We added that entry in
`3dfdc8b4a5` during the Block-1 experiment; we then deleted the directory it points at in
`9251fa6026` ("park the Block-1 demo out of the template pool") without removing the entry.

Parking the directory alone does not park the backend: selection still ranks and picks it (RUN_REPORT
15:11), and it is now a template with no template. Either drop the registry entry or restore the
directory — and never leave a local absolute path in a shared registry.

---

## Corrections to this document

- **`beat_baseline: false` on 24/24 kernel records is BY DESIGN, not a defect.** I flagged it as a
  possible reporting bug before reading `perf_mcp.py:4430`, which stores the agent's argument as
  `claimed_beat_baseline` and pins `beat_baseline` to `False` unconditionally; `_ledger().is_win`
  owns the verdict from measurements. The comments at 3941 and 4412 record that trusting the
  agent's flag previously double-counted wins. This is the same anti-self-certification design as
  F6's tool split, and it is correct.
- **The optimizer's state was tracking correctly.** I earlier read a `state.json` showing
  `iteration: 0` and reported a reporting gap. That file is not the optimize ledger; the live state
  is `/tmp/perf_mcp_*_voxtral_tts_backbone_main.json`, which tracked every commit accurately. No
  defect.
- **"Voxtral and XTTS are not on HuggingFace" is wrong** where it appears above. Both are on the
  Hub; they ship under `library=vllm` and `library=coqui` respectively, in their own native
  formats, not in `transformers` format. The tool's loader requires the latter — that is the actual
  constraint, and S1 is the consequence of it.

---

## Observations (not defects — recorded for the comparison write-up)

**O1 — whole-model dtype only.** `plan` recommends "N150 with bfp8_b weights", one dtype for the
whole model. The hand-port's §6.16 measured per-weight precision as the deciding factor: BFP8 on
FF and attention but **w2 in bf16**, because w2 alone is 77% of the accuracy cost for 15% of the
speed. There is no per-weight axis in the recommendation.

**O2 — a real static catch, worth crediting.** Compat flagged that this config uses HF's newer
`rope_parameters` field while `tt_transformers/tt/model_config.py:2736` only reads `rope_scaling`,
so the runtime would **silently** treat the model as having no scaling — safe at short context,
divergent at long. Found before anything ran, with two concrete fixes offered.

**O3 — confident tone on a fallback path.** `plan` prints `CONFIDENCE: HIGH` while also stating
"weights-only estimate (no transformer config); no KV math applied". The KV term is genuinely
omitted; here it does not change the verdict (218 MB at max_seq_len 2048 against ~24 GB headroom),
but the label and the caveat disagree.

---

## Process notes (mine, recorded so the numbers above can be judged)

Three mistakes cost real time and one of them destroyed evidence:

- **Trusted `pgrep` three times.** It matches any process whose command line contains the search
  text, including the lingering shell wrappers of finished jobs. It reported dead runs as alive
  and a live run as dead. `ps` is the reliable check.
- **Piped two long jobs through `tail`.** `tail` buffers until the process exits, so the build's
  CMake error was lost and the 9-hour overnight run wrote **nothing** to its log. Everything about
  that run had to be reconstructed from the tool's own state files and file timestamps. Use
  `python -u` and a direct redirect.
- **Reported progress from a log line instead of a verified state**, and called a run "past the
  wall" moments before it died at the same place.
- **Committed findings to the branch the tool was optimizing on.** The tool stamps the current HEAD
  into its own measurement records, so `full_pipeline_baseline` for the 15.976 result carries the
  sha of a *documentation* commit. Harmless here, and the agent noticed unprompted (*"the extra HEAD
  commit is the harness's own findings entry sitting on top of my perf commit"*), but the findings
  should have lived on a separate branch. `git log --grep '^perf('` isolates the tool's commits.

All PCC numbers in this file come from a direct `pytest` run I executed, not from the tool's
status files.
