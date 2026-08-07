# advchal-v2 — how the 15 cells captured, and why it is not one experiment

The capture is the advisor's **only** input (see [`FINDINGS`](ADVCHAL-V2-FINDINGS.md) §3.5). What a capture
script does therefore bounds what the advice can possibly be, and what the cell can possibly measure.

Fifteen cells wrote fifteen capture scripts from one template. **Nothing in the skill or the gate compares
them**, so differences in capture scope read downstream as differences in the advisor's usefulness. This file is
the comparison.

---

## The scripts, side by side

| cell | file | lines | traces `decode_forward`? | model methods substituted | private env knob |
|---|---|---|---|---|---|
| llama-3.1-8B exp17 | `capture.py` | 54 | yes | — | `CHALLENGER_FINALIZE_CAPTURE` |
| llama-3.2-1B exp17 | `capture.py` | 92 | yes | — | — |
| phi-3.5 exp17 | `capture.py` | 102 | yes | — | — |
| gemma-4-12B exp11 | `capture.py` | 113 | **no** | — | — |
| north-mini FN | `capture.py` | 120 | **no** | — | — |
| phi-3.5 onA | `capture.py` | 129 | yes | **`_decode_rope`** | — |
| north-mini onA | `capture.py` | 135 | yes | — | **`CHALLENGER_CAPTURE_ATTENTION_ONLY`** |
| gemma-4-26B onA | `capture_advisor_challenger.py` ¹ | 143 | **no** | — | — |
| gemma-4-26B FN | `capture_advisor_challenger.py` ¹ | 150 | **no** | — | — |
| north-mini B | `capture.py` | 154 | yes | — | — |
| phi-3.5 B | `capture.py` | 156 | **no** | **`_decode_rope`** | — |
| qwen3-27B B | `capture.py` | 178 | yes | **`_rms_norm_decode`, `_decode_linear`, `_partial_rope_decode`** | — |
| phi-3.5 FN | `capture.py` | 219 | yes | **`_decode_rope`** | — |
| qwen3-27B FN | `capture.py` | 290 | yes | — | — |
| gemma-4-26B B | `capture_advisor_challenger.py` ¹ | 84 | **no** | — | — |

¹ The three gemma-4-26B cells also put the script in `models/autoports/<model>/tests/` rather than
`doc/advisor_challenger/`, so it is not alongside the artefacts it produces.

**54 to 290 lines for the same job.** That spread is the headline: these are not fifteen runs of one procedure.

---

## Axis 1 — four cells substitute model methods before tracing

The most consequential difference, because the advisor then reasons about code the model does not run.

| cell | substituted | why, per the script |
|---|---|---|
| phi-3.5 FN, B, onA | `_decode_rope` | *"the capture template forbids dynamic `tensor.memory_config()` queries: layout is the optimizer's to assign"* — so each writes a stand-in with the declared L1 height-sharded config |
| **qwen3-27B B** | **`_rms_norm_decode`, `_decode_linear`, `_partial_rope_decode`** | three methods, no stated reason in the script |

**phi:** three of four arms replace the RoPE body. The reason is real — the tracer cannot resolve
`memory_config()` before layout assignment — but the consequence is that **the advisor never sees the shipped
RoPE**, and its advice for that region is advice for a substitute. phi exp17 is the one arm that does not
substitute, and it traces the real method.

**qwen B is the extreme case.** It replaces its normalization, its linear, and its rope — **three of the op
classes this corpus's findings are about**. Any statement about what the advisor recommended for qwen B's norms
or linears is a statement about the stand-ins, not the decoder.

*(This refines a claim made elsewhere in the corpus: the RoPE substitution is **not** a property of the capture
template. It is a choice three phi cells made, and one did not.)*

---

## Axis 2 — where the trace stops at a terminal

`ttnn.sparse_matmul` is terminal in the emit tracer for every MoE model here, so five cells faced the same wall.
They stopped in four different places:

| cell | what it did | ops captured |
|---|---|---|
| **gemma-4-26B FN, onA** | substitute the **dense** expert path — `_DECODER._dense_mlp(...)` — and trace on through the post-FF norm. This is the remedy the tracer's own docstring prescribes | **29 / 25**, **30 / 26** |
| north-mini B | truncate inside its own `decode()`, in-script, no env knob | 16 / 18 |
| north-mini onA | private env knob, truncate at the **attention** boundary | 14 / 16 |
| north-mini FN | nothing specific, and also lost SDPA, concat-heads, paged cache, `topk`, `scatter` | **5 / 7** |

All five declared the op in `report.json`'s `uncapturable` field, which is the supported mechanism, correctly.
**The variance is entirely in where they chose to stop** — and it is a 6× spread in captured ops for the same
model family and the same wall.

---

## Axis 3 — six of fifteen never trace the model's own `decode_forward`

gemma-4-12B, north-mini FN, phi-3.5 B and all three gemma-4-26B cells hand-write the traced path instead. That
is sometimes necessary — it is how gemma-4-26B reaches past the sparse terminal — but it means the traced graph
is the cell author's reconstruction of a decode step, and nothing checks it against the real one.

The one check that would catch a divergence — **compare the captured op sequence against the profile's device
rows** — is exactly what `reconcile.py` does, and its `untraced` share is the signal. But `untraced` is reported
as a *coverage* number, never as a *fidelity* number, so a hand-written path that quietly omits an op looks
identical to a tracer terminal.

---

## Axis 4 — two cells invented private environment knobs

| knob | cell | in the template? |
|---|---|---|
| `CHALLENGER_CAPTURE_ATTENTION_ONLY` | north-mini onA | no |
| `CHALLENGER_FINALIZE_CAPTURE` | llama-3.1-8B exp17 | no |

Neither is wrong in itself. Both are invisible to anyone reading the artefacts, because the value used is not
recorded in `report.json` — so a reader cannot tell which mode produced the capture they are looking at.

---

## What this costs, and the three cheap fixes

**It costs comparability.** The corpus's cross-cell numbers — untraced shares, advised-op counts, hit rates —
mix cells whose captures attempted very different amounts of the layer. A 14-op capture and a 30-op capture of
the same model family are not the same evidence, and nothing in the artefacts says so.

1. **Record the capture's own scope in `report.json`** — ops attempted, methods substituted, env knobs and their
   values. One dict. It makes every cross-cell comparison auditable, and it is the single highest-value change
   in this file.
2. **Put the terminal recipes in the skill.** The dense-expert-graph substitution works and is documented in
   `ttnn_emit_tracer`'s docstring; two cells found it and three did not. A cell meeting a terminal should not be
   inventing a private knob.
3. **Give the template a supported hook for the `memory_config()` restriction**, which is what drives the phi
   RoPE substitution. Three cells wrote three versions of the same workaround.

→ [`FINDINGS`](ADVCHAL-V2-FINDINGS.md) §3.5 for what the tracer is and what it drops,
[`STAGE-ANALYSIS`](ADVCHAL-V2-STAGE-ANALYSIS.md) D13 for the capture defect,
[`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md) for the action list.
