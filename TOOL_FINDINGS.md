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

All PCC numbers in this file come from a direct `pytest` run I executed, not from the tool's
status files.
