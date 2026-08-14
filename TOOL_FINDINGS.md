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

**If only one thing is taken: F6.** It is the difference between "this tool does not work" and
"this tool ported a 3.4B model correctly in ten minutes".

### B. OUR SIDE — not tool defects, do NOT report these

| # | what | whose fault | note |
|---|---|---|---|
| **S1** | the HF export shipped no tokenizer | **ours** | our exporter predates this use; fixed by converting `tekken.json` (15/15 vs ground truth) |
| **S2** | `tt-perf-report` not installed → `optimize` preflight failed | **ours** | it **is** declared in `requirements-agent.txt`; we simply never ran that install. The tool detected it, refused cleanly, and named the fix — correct behaviour |
| **S3** | the model had to be converted to HF format at all | **ours / inherent** | Voxtral ships in Mistral-native format; every model this tool handles arrives as a `transformers` model. Not a defect |

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
| **run 2** — Q+K rotated in one call (O4f) | 269.5 | **14.909** | 0.9903 |
| tool's own roofline target | 338.541 | — | gate 0.95 |
| **hand-port, for reference** | — | **15.907** | — |

**14.909 against 15.907 — the tool is 6.3% AHEAD of 74 human experiments**, autonomously, holding
PCC 0.9903.

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
