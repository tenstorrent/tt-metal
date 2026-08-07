# advchal-v2 — how the 15 cells captured, and why it is not one experiment

The capture is the advisor's **only** input (see [`FINDINGS`](ADVCHAL-V2-FINDINGS.md) §3.5). What a capture
script does therefore bounds what the advice can possibly be, and what the cell can possibly measure.

**A note on what is authoritative here.** The skills, `ttnn-jit` and the capture scripts are all young code; a
comment in them explains what the code does, not necessarily a considered design. Where this file quotes them —
the `--tracer` help, the "loud fail" rationale — treat it as *what happens*, not as *what was intended*. The
tt-mlir optimizer is the stable component, and claims about it (`OpModelExempt`, `LayoutScore`) are read from its
source.

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

## Axis 3 — six of fifteen do not call `decode_forward`. Measured, it costs nothing

First, what the choice actually is. **The advisor is always called on the traced graph** — there is no
compiled-versus-hand-written distinction. Whatever Python the capture's `decode()` executes is what gets traced.
The variance is only in *what that function executes*:

| pattern | cells | what `decode()` does |
|---|---|---|
| calls `decode_forward` | 9 | one call into the model's own decode step |
| **transcribes it** | 5 — north-mini FN, phi-3.5 B, gemma-4-26B ×3 | replays the decode step, mostly through the model's **own** sub-methods (`self._norm_decode`, `self._linear_decode`, `self._decode_rope`, `_DECODER._dense_mlp`) |
| shared helper | 1 — gemma-4-12B | delegates to a `common.decode` in the cell's own tree |

"Transcribes" is the accurate word, not "hand-writes a graph". phi B's version calls seven of the model's own
methods and interleaves eleven `ttnn` calls; gemma-4-26B FN's calls nine. These are replays of the shipped path,
not independent reconstructions.

### Why they do not all just call `decode_forward` — tested, and the blocker is real

**Because that call is all-or-nothing.** At the first unhandled op the trace raises out of `trace_ttnn`, through
`cli.py`, to exit 1: **no IR, no report, no advice, for the whole layer.** Verified by re-running north-mini onA's
full `decode_forward` with its truncation flag off — exit 1, and the only artefact is a traceback.

Transcribing lets a cell stop immediately before the wall and keep everything up to it. Every transcribing cell
says so in a comment:

> *"The pinned direct tracer cannot consume TracedTensor inputs in `paged_fused_update_cache`, and
> `sparse_matmul` is terminal by contract. Preserve the real shipped path up to those tracer boundaries."*
> — north-mini FN

**Is there an escape hatch?** `ttnn-advise capture` takes `--tracer {ttnn,interception,rewrite}`, default `ttnn`,
whose help says: *"Use **interception** only for ops not yet handled by the direct-TTNN tracer."* That is exactly
this situation, `interception_tracer` **does** have a `sparse_matmul` handler — and **no cell ever passed
`--tracer`.**

**So I tried it. It does not work either, and for a different reason.** With `--tracer interception` on
north-mini's full decode, the trace dies *earlier* — in `_qkv_decode`, on
`ttnn.experimental.rotary_embedding_hf`, which the interception tracer does not handle. **The two tracers have
disjoint gaps:**

| op north-mini needs | emit tracer (`ttnn`) | interception tracer |
|---|---|---|
| `sparse_matmul` | **missing** | present |
| `rotary_embedding_hf` | present | **missing** |
| `paged_fused_update_cache` | **missing** | **missing** |
| `ones_like` | **missing** | **missing** |
| `topk`, `scatter` | present | present |

**No tracer choice captures north-mini's decode.** Emit dies at the experts, interception dies before the
attention even finishes, and both lack the paged-cache update that phi and north-mini FN also cite. **The
truncation was not avoidable** — a full capture of this layer is not available at this pin by any documented
route.

That corrects the implication elsewhere in this corpus that cells truncated out of haste. What *was* avoidable is
how much they kept: gemma-4-26B's dense-expert substitution stays inside the emit tracer's coverage and captures
30 ops where north-mini onA's attention-boundary cut captures 14. **Getting more was possible; getting
everything was not.**

**Whose problem this is.** These are `ttnn-jit` coverage gaps — missing per-op handlers in two tracers whose
coverage has drifted apart. Nothing about it implicates the tt-mlir optimizer, and adding the handlers is
additive work rather than a design change.

### The gap was one handler, and porting it worked

`ttnn-jit` lives in **`tt-mlir/tools/ttnn-jit/`** — first commit 2025-09-19, 127 commits, last touched
2026-07-27. Real upstream tooling, young and still filling in.

**The two tracers reach coverage differently, and that is the whole story.** `supported_ops.py` defines an
allowlist of **54** ops in six categories (unary 25, binary 16, reduction 4, tm 4, data-movement 3, ccl 3), plus
`matmul`/`div`/`pow`. The TTIR tracer routes every allowlisted op through one generic `BaseOpHandler`, so it gets
all 54 for free and needs only **27** hand-written handlers on top. The direct-TTNN tracer has no generic path:
it needs a handler per op and stubs anything else via `_unhandled`. It carries **85**.

Counting them properly — explicit handlers for the emit tracer, explicit **plus** the allowlist for the
interception tracer:

| direction | count | ops |
|---|---|---|
| **interception reaches, emit did not** | 4 | **`sparse_matmul`**, and `pow` / `pow_tensor` / `rearrange` (allowlisted, so `_unhandled` stubs) |
| **emit reaches, interception does not** | 9 | `arange`, `fill_cache`, `logical_and`, `mul`, `pad`, `rotary_embedding`, **`rotary_embedding_hf`**, `scaled_dot_product_attention_decode`, `zeros` |
| **neither** | 2 | `paged_fused_update_cache`, `ones_like` |

So for north-mini's sparse-MoE decode, **`sparse_matmul` was the only missing piece** — router (`topk`, `scatter`,
`zeros_like`, `sigmoid`), experts, and the MoE tail were all already covered by the emit tracer. And the
documented fallback fails for a narrower reason than a coverage deficit: `--tracer interception` is offered "for
ops not yet handled by the direct-TTNN tracer", but `rotary_embedding_hf` is neither allowlisted nor hand-written
there, so it dies on RoPE before reaching the sparse tail. One missing handler each way.

### The port, and what it bought

**The change:** one 37-line handler transcribed from `interception_tracer._sparse_matmul_handler`, one line in
`_VALUE_HANDLERS`, and a docstring correction. 42 inserted lines, no C++ and no rebuild — the tracers are pure
Python.

**Result — the same cell, the same profile window (822.608 µs, one layer), the same advisor pin:**

| | shipped (14 traced ops) | ported (39 traced ops) |
|---|---|---|
| untraced | 32 ops, **77.15 %** | 16 ops, **14.39 %** |
| chain (screenable disagreement) | 4 ops, 12.41 % | 14 ops, **23.45 %** |
| `agrees_with_shipped` | 1 op, 3.16 % | 3 ops, 6.24 % |
| `dram_resident` | 2 ops, 1.26 % | 6 ops, **49.90 %** |
| boundary | 8 ops, 6.02 % | 8 ops, 6.02 % |
| chains found | 3 | 8 |
| **ceiling vs noise floor** | 0.562 µs = **0.66×** | 8.543 µs = **10.09×** |
| **chains resolvable alone** | **0** | **2** |

The advisor went from seeing 22.8 % of the layer to 85.6 % of it. The cell published a zero and was *right* to —
its ceiling was below the noise floor, so there was nothing to screen. With the handler in place there are two
candidates above the floor: a `scatter` chain worth 2.825 µs (3.34× floor, **33.9 µs/model**) and a boundary chain
`b24` worth 2.331 µs (2.75× floor, **28.0 µs/model**). Neither existed before, and neither is in the corpus's
reachable total.

**And it prices the `OpModelExempt` gap.** The advisor now sees both `sparse_matmul` ops and leaves them at
`dram/interleaved` — it places nothing on them, exactly as `OpConstraintValidation.cpp:167-177` says it will. Those
two ops are **47.4 %** of the window on their own (241.506 + 148.461 µs of 822.608). That is the soft gap made
concrete: not a crash, but half a layer the optimizer declines to reason about — while still placing the router
and tail around it, which is the reason to trace it anyway.

### How hard was it, really

**Easy, and for reasons that will not generalise evenly.** What made it a transcription rather than a design task:

- the shape logic already existed in the sibling tracer, with the three sparse modes worked out;
- `TTNN_SparseMatmulOp` and its Python binding already existed, with the same operand and attribute names as
  the TTIR op — so the diff was `ttir.sparse_matmul` → `ttnn.sparse_matmul` plus `_retype` for the layout'd
  result type;
- `SparseMatmulOp::verify` (`TTNNOps.cpp:2720`) states the output-shape relation explicitly in its error
  branches, and it matched the TTIR handler's arithmetic exactly — so the port could be **checked** rather than
  guessed;
- `_capture` already passes scalars, `None` and non-tensor objects through, so `memory_config`, `output_tile`,
  `program_config` and `dtype` from the real call site needed no handling at all.

The judgement calls were small but real: drop `program_config`/`compute_config` (every direct-TTNN tensor starts
DRAM-interleaved and the optimizer re-decides, so a carried-over config would be a lie), default to b-sparse when
neither flag is set (matching ttnn), and trust the layout builder with a rank-6 result — which it accepted.

Roughly ten minutes of reading, five of editing, 40 seconds to run.

### Could a skill do this automatically?

Yes for the class this belonged to, and it is worth building, but the class is nearly exhausted in this direction.

**What a skill can do mechanically.** On a `_unhandled` raise or a `cannot consume ... TracedTensor` TypeError,
name the op, then:

1. **Port from the sibling tracer** if it has a handler — a two-token substitution plus `_retype`. After this
   port, that leaves `pow`/`pow_tensor`/`rearrange`, which the allowlist generator below covers anyway.
2. **Generate from the allowlist** for anything in `supported_ops.py` with no emit handler: the shape rule is
   fixed per category (unary/binary/reduction/tm/data-movement), which is exactly what `BaseOpHandler` relies on.
3. **Derive from the verifier** for the genuinely new ops. This is the interesting one, and the
   `SparseMatmulOp::verify` case suggests it is tractable: the verifier spells out the shape relation in prose
   *and* in comparisons (`outputShape[0] != A || outputShape[1] != B || ...`), so it can be read and inverted
   into a shape function rather than inferred from `TTNNOps.td`, which carries only `AnyRankedTensor`.

**Why it is safe to automate: there is a hard oracle, and it is cheap.** A wrong shape does not produce a subtly
bad capture — the MLIR verifier rejects the op immediately. So a skill can propose a handler, trace a
single-op function, and know within seconds whether it is right. Where the verifier is only a rank check, compare
against the real `ttnn` op's output shape at the live call site, which is right there in the model.

**What it cannot decide.** Which non-shape kwargs to drop, and whether dropping them is honest for the pipeline
being run. Here that meant knowing the advisor discards input memory configs — a fact about the *advisor*, not
the op. That belongs in the skill's instructions, not in its inference.

### And the measurement: transcription does not cost coverage

Untraced share per cell/kind, against what the capture did:

| | kinds | untraced share |
|---|---|---|
| **a terminal was declared** | 14 | **38.8 % – 77.2 %** |
| **no terminal** | 12 | **2.1 % – 14.8 %** |

**Complete separation, no overlap.** Coverage is set by terminals, not by capture style. Within the no-terminal
group the approach barely registers:

| approach | kinds | mean untraced | range |
|---|---|---|---|
| shared helper | 2 | 3.5 % | 2.1 – 4.8 |
| calls `decode_forward` | 9 | 7.5 % | 2.5 – 14.8 |
| transcribed | 1 | 10.7 % | — |

The cleanest comparison is **phi, four arms of one model, no terminals**: exp17 **9.1 %** (`decode_forward`),
B **10.7 %** (transcribed), FN **13.4 %** (`decode_forward`), onA **14.8 %** (`decode_forward`). The transcribing
arm sits mid-range and beats two of the three that called `decode_forward`.

**So less hand-writing would not give more useful results.** The lever is the terminals — port the ops, or use
the documented dense-expert substitution — not the writing style. Two caveats worth keeping:

- Cross-model untraced shares are **not comparable** (a model whose MoE tail is a larger fraction of its layer
  will show a larger share for the same wall), so the terminal-versus-no-terminal split above is the reliable
  cut, not the ordering inside it.
- The real risk of transcription is **silent drift** from the shipped path, and nothing checks for it. phi B
  shows it need not materialise; it also means nobody would know if it had. That is the argument for recording
  capture scope, not for banning transcription.

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
