# Every blocker this corpus recorded: which are real, which are fixable, and what fixing them bought

Seven of fifteen cells returned zero, and the docs attribute three of those to "coverage gaps" — layers the
advisor never saw. This audits every op those cells named as a blocker, decides what class of fix each needs, and
where a fix was cheap, applies it and re-measures.

**Headline:** of the ten ops cells named as blockers, **four were already handled** and **one does not exist**.
Two real blockers were never named at all. Every real one is now fixed — 218 lines of pure Python across the two
tracers, no rebuild — and coverage on eight cell/kinds went from 58–77 % untraced to 4–21 %. One cell that
published a flat zero now has **11 screenable candidates worth 632 µs/model**.

**The code is on a tt-mlir branch:** [`mvasiljevic/ttnn-jit-tracer-coverage-gaps`](https://github.com/tenstorrent/tt-mlir/tree/mvasiljevic/ttnn-jit-tracer-coverage-gaps) — one commit,
`756e134a1b`, based on the corpus's own advisor pin `618cd4e75d`. Two files, 218 insertions, pure Python, no rebuild.

Method: re-capture each cell with the handler added, then re-run `reconcile.py` against **the cell's own profile
window and incumbent**, with the shipped report re-run through the identical command as a verified control. Where
the control did not reproduce the cell's published numbers, the comparison is not shown.

---

## 1. The complete blocker list, swept from the artefacts

Not from what the docs mention — from every `uncapturable.ops` entry in all 15 cells' `report.json`:

| op, as a cell recorded it | cell/kinds | real? | what a fix needs | status |
|---|---|---|---|---|
| `ttnn.sparse_matmul` / `sparse_matmul` | **12** | yes | port the handler the sibling tracer already has | **done, measured** |
| `paged_fused_update_cache` | 3 | yes | it *is* `(cache1, in1, cache2, in2, …)` — two `paged_update_cache` ops | **done, verified in IR** |
| `paged_scaled_dot_product_attention_decode` | 3 | **NO — already handled** | — | see §3 |
| `nlp_concat_heads_decode` | 3 | **NO — already handled** | — | see §3 |
| `topk` | 2 | **NO — already handled** | — | see §3 |
| `scatter` | 2 | **NO — already handled** | — | see §3 |
| `ttnn.ones_like` | 2 | yes | `ttnn.ones` at the input shape — a copy of `_zeros_like_handler` | **done, measured** |
| `ttnn.copy` | 1 | yes | tracer bookkeeping only; no dialect op exists or is needed | **done, traces** |
| `ttnn.softplus` | 1 | yes | no standalone dialect op → 3-op decomposition | **done, traces** |
| `ttnn.recurrent_state_update` | 1 | **NO — the op does not exist** | — | see §4 |

Two more were never recorded by any cell and had to be found by re-running:

| found by re-running | real? | what a fix needs | status |
|---|---|---|---|
| **`TracedTensor.__getitem__`** | yes | **not an op** — a proxy-protocol gap | **done, traces** |
| `ttnn.repeat_interleave` | yes | `TTNN_RepeatInterleaveOp` exists | **done, oracle-tested** |

And three that no cell hit but the tracers still lack:

| gap | what a fix needs | status |
|---|---|---|
| `pow` / `pow_tensor` | `TTNN_PowTensorOp` / `TTNN_PowScalarOp` exist | **done, oracle-tested** |
| `rearrange` | no dialect op; the einops pattern string needs parsing, then `permute` + `reshape` | not done — no cell needs it |
| `rotary_embedding_hf` *in the interception tracer* | **cannot be ported.** TTIR has only `rotary_embedding_llama`, which requires a `trans_mat` operand the HF op has no equivalent for. Fabricating one would add a tensor the advisor then places | **drop the `--help` fallback claim instead** |

**So of the ten ops cells recorded as blockers, four were already handled and one does not exist.** Five were
real, and all five are now fixed. Nothing in the list is a tt-mlir optimizer defect.

## 2. What the fixes bought, measured

Each row is the same cell, the same profile window, the same advisor pin (`618cd4e75d`), with the shipped report
re-reconciled through the identical command as a control.

| cell / kind | untraced, shipped → fixed | screening ceiling vs noise floor | chains resolvable alone |
|---|---|---|---|
| north-mini onA, full sparse MoE | **77.15 % → 14.39 %** | 0.66× → **10.09×** | **0 → 2** |
| north-mini onA, sliding sparse MoE | **76.62 % → 15.47 %** | 0.12× → 0.67× | 0 → 0 |
| gemma-4-26B onA, full attention | **58.51 % → 3.86 %** | 1.70× → 3.29× | 0 → 0 |
| gemma-4-26B onA, sliding attention | **64.70 % → 4.17 %** | 4.08× → **13.74×** | **2 → 4** |
| **north-mini B, sliding rope MoE** | **75.66 % → 21.12 %** | 1.80× → **22.77×** | **0 → 6** |
| **north-mini B, full no-rope MoE** | **76.40 % → 21.48 %** | 0.45× → **16.09×** | **0 → 5** |
| north-mini FN, full attention MoE | 5 → **37 ops captured** | *no comparison — see below* | — |
| north-mini FN, sliding attention MoE | 7 → **39 ops captured** | *no comparison — see below* | — |
| qwen3.6-27B B, linear attention | trace aborted → **71 ops captured** | *see §5* | *see §5* |

**north-mini B is the largest result in this audit.** The cell published a flat zero — *"all measured geometries
slower or stalled"* — with a screening ceiling of 0.45× and 1.80× its own noise floor. Two handlers later
(`ones_like` and `sparse_matmul`) it has **11 candidates above the floor worth 632.1 µs/model between them**,
which is more than every shipped win in the corpus combined:

| kind | chain | removes | vs floor | per model |
|---|---|---|---|---|
| sliding rope MoE | `:5` `scatter` | 3.367 µs | 3.60× | **121.2 µs** |
| sliding rope MoE | `:4` `scatter` | 2.825 µs | 3.02× | **101.7 µs** |
| sliding rope MoE | `:b32`, `:b40`, `:b37`, `:b33` | 2.339 / 2.321 / 1.475 / 1.107 µs | 2.50–1.18× | 84.2 / 83.6 / 53.1 / 39.9 µs |
| full no-rope MoE | `:5` `scatter` | 3.382 µs | 2.68× | **40.6 µs** |
| full no-rope MoE | `:4` `scatter` | 2.848 µs | 2.25× | **34.2 µs** |
| full no-rope MoE | `:b36`, `:b28`, `:b33` | 2.359 / 2.305 / 1.481 µs | 1.87–1.17× | 28.3 / 27.7 / 17.8 µs |

*(Per-cell figures. Do not add across arms of the same model — B and onA are alternatives, not additive.)*

**north-mini FN's captures were the most truncated in the corpus and are shown without a reconciliation.** Its
sparse branch returned `query` — it stopped immediately after QKV, before attention, before the cache update,
before the experts. With `paged_fused_update_cache` and `sparse_matmul` handled it captures 37 and 39 ops. But no
profile CSV in that branch reproduces the cell's published 67.9 % / 69.0 % untraced share, so the accounting
comparison is **not shown** rather than shown against a different window. The op counts are the finding.

The fused-cache decomposition is verified in the IR: `paged_fused_update_cache` emits exactly
`2 "ttnn.paged_update_cache"`.

**Four new screenable candidates, none of them in the corpus's reachable total:**

| cell / kind | chain | removes | vs floor | per model |
|---|---|---|---|---|
| north-mini onA, full MoE | `:3` `scatter` | 2.825 µs | 3.34× | **33.9 µs** |
| north-mini onA, full MoE | `:b24` boundary | 2.331 µs | 2.75× | **28.0 µs** |
| gemma-4-26B onA, sliding | `:5` `scatter` | 2.834 µs | 4.83× | **70.8 µs** |
| gemma-4-26B onA, sliding | `:6` `multiply` | 2.834 µs | 4.83× | **70.8 µs** |

**Two cells gained coverage but still cannot screen**, and that is worth stating plainly: north-mini's sliding
kind and gemma's full-attention kind have noise floors of ~14.5 µs and ~3.5 µs against ceilings of 9.7 µs and
11.7 µs. Coverage was never their binding constraint — harness noise is. Fixing the tracer does not fix that, and
the per-process floor work (STG-5 / I5) is what would.

**And the `OpModelExempt` gap gets priced, twice.** With the experts visible, `dram_resident` — ops the advisor
sees and declines to place — jumps to **49.90 %** (north-mini full), **48.87 %** (north-mini sliding),
**40.08 %** (gemma full) and **44.29 %** (gemma sliding) of each window. On north-mini the two `sparse_matmul`
ops alone are **47.4 %** (241.506 + 148.461 µs of 822.608). That is not a crash and not a regression: it is half
of each layer that the optimizer will not reason about, while it does place the router and the tail around it.
Tracing the op is still what makes the surrounding placement possible.

---

## 3. Four recorded blockers were already handled

`paged_scaled_dot_product_attention_decode`, `nlp_concat_heads_decode`, `topk` and `scatter` are all registered in
the direct-TTNN tracer — `_TRANSFORMER_VALUE`, `_EXPERIMENTAL_VALUE` and `_VALUE_HANDLERS` respectively. The
re-runs prove it: north-mini FN's new capture contains `paged_scaled_dot_product_attention_decode` and
`nlp_concat_heads_decode`, and every MoE capture contains `topk` and `scatter`.

**How the mistake happened is worth knowing, because it recurs.** `nlp_concat_heads_decode` and
`rotary_embedding` dominate the corpus's `unfixable_ops` field — 20 and 18 declarations. That field means *the
advisor's constraint query failed, so it will not place this op*. It does **not** mean the tracer cannot trace it.
Two cells copied names from one into the other, turning a placement limitation into a coverage claim, and the
tracer's own docstring made the same error in the other direction — it declared the MoE router
(`topk`/`scatter`/`zeros`/`arange`/`pad`/`clamp`/`fill_cache`) unsupported long after handlers were added.

**`uncapturable.ops` and `unfixable_ops` answer different questions.** Anything written into the first should be
reproduced by an actual trace attempt first.

## 4. `ttnn.recurrent_state_update` does not exist

north-mini's and qwen's reports name three terminal ops for the linear-attention kind:

```json
"ops": ["ttnn.copy", "ttnn.softplus", "ttnn.recurrent_state_update"]
```

The third is not a TTNN op, not a `ttnn` Python attribute, and appears nowhere in the model. The state write is a
`ttnn.copy`:

```python
ttnn.copy(recurrent, self.caches["recurrent"])          # optimized_decoder.py:1274
```

`recurrent_state_update` occurs in exactly two files corpus-wide — the cell's own `report.json` and its
`reconciliation_linear_attention.json`. It is a **label the cell invented**, most likely by reading a device-op
name out of the profile and writing it back as a ttnn op. It made the blocker list look one item longer and one
item more exotic than it was.

**For anyone reading a `uncapturable.ops` list: check each name resolves to a real `ttnn` attribute.** Two of
qwen's three did.

---

## 5. qwen: the coverage gap is closed, and the failure moved

qwen is the biggest prize in the corpus — `linear_attention` is **97 % of its model decode time** and 63.5 % of
that window was untraced, so ≈62 % of the model was never advised on. Its untraced bulk is not exotic:

| device op | µs | share of window |
|---|---|---|
| `MatmulDeviceOperation b={1536} x 32 x 128 x 128` | 3435.918 | 21.70 % |
| `PermuteDeviceOperation` | 1965.426 | 12.41 % |
| `BinaryNgDeviceOperation` | 1943.759 | 12.28 % |
| `SliceDeviceOperation` | 1595.180 | 10.08 % |

Ordinary matmul, permute, binary and slice — all of which the tracer already handles. They were untraced only
because the trace **aborted upstream** of them.

**Four gaps, found one at a time by re-running:**

1. `ttnn.copy` — no dialect op. Modelled as an identity alias: rebind the destination's identity to the source's
   traced value, emit nothing. Needs the raw caller object, so it patches before argument capture.
2. `ttnn.softplus` — no standalone dialect op (`SoftPlus` exists only as a `UnaryOpType` enum case usable as a
   matmul fused activation). Decomposed to `log(exp(x) + 1)`. **Three ops for one** — faithful in value, not in
   op count, and flagged in the handler docstring.
3. **`TracedTensor.__getitem__`** — the mixer splits its projection with `mixed[..., :key_width]`. Plain Python
   subscript syntax, no op involved. The trace died on `'TracedTensor' object is not subscriptable`, which names
   no op and appears in no blocker list. Implemented by routing to the *patched* `ttnn.slice`, so it works for
   either tracer. An integer index would drop a rank, which `ttnn.slice` cannot do, so that raises rather than
   silently keeping the axis.
4. `ttnn.repeat_interleave` — `TTNN_RepeatInterleaveOp` exists; a 12-line handler.

**With all four in, the trace completes: 71 ops**, including `repeat_interleave` ×2, the `exp`/`ones`/`log` triple
from the softplus decomposition, `slice_static` ×8 from the subscripts, and the mixer's `matmul`, `permute`,
`sum`, `silu`, `sigmoid` and `rms_norm`. Verified by tracing directly and printing the module.

**And the failure after that was mine, not the optimizer's.** `ttnn-advise` aborted inside
`mlir::PassManager::run` with no MLIR diagnostic — which looks like an optimizer crash. Captured unfiltered, the
message is:

```
LLVM ERROR: Backend constraints are not implemented for op ttir.empty
```

Exactly **two** `ttir.empty` ops survived into the TTNN module, of shapes `1x32x10240x4` and `32x48x128x128` —
qwen's two state caches, which are the two `ttnn.copy` destinations. My `copy` handler recorded its rebinding in
`jit_ctx.weight_cache`, and a destination usually already **has** a placeholder there from an earlier read (the
mixer reads the previous state before writing the new one). Overwriting the entry orphaned that placeholder, so
`_finalize_signature` — which lifts and erases placeholders by walking `weight_cache` — never saw it.

The fix records the rebinding in `jit_ctx.cache_alias` and has `_weight_value` consult `cache_alias` before
`weight_cache`, which is what `interception_tracer._weight_value` already did. **With that, the layer captures:
69 ops advised, `uncapturable: none`**, with no regression on the captures that already worked.

**Capture unblocked; the gain is not quantified.** With the fifth handler in, qwen's whole layer captures — **69 ops advised, `uncapturable: none`**. But the cell never kept its own profile (STG-10), and no committed CSV reproduces its 15,833 µs window, so there is no sound before/after accounting. Reconciling against a different window produces figures that are visibly wrong — an unchanged `boundary` bucket, `untraced` barely moving despite 3.4× the ops, and a "resolvable" total larger than the model — which is the positional-pairing hazard (STG-7) doing exactly what it is documented to do. **The reachable value here needs a fresh profile, not more analysis.**

What had been established before the fix, and is worth keeping as method:

- each new handler passes a one-op trace in isolation (§6), so none of them emits invalid IR on its own;
- the traced module verifies and parses — `ttmlir-opt --ttnn-to-ttnn-l1-advisor` accepts it and exits 0;
- the crash was inside the pass pipeline, in-process.

All true, and none of it located the bug. What did was **removing the log filter**: the abort message had been in
the output all along, buried under 83,000 lines of routine `TT_FATAL` constraint-query rejections, and I was
grepping for `Traceback` and `error:` — which an `LLVM ERROR` line matches neither of. **The cheapest diagnostic
step was the one I skipped: read the tail of the log unfiltered.**

It does matter that I wrote *"not established that this is an optimizer defect"* rather than blaming the optimizer.
That was the right call: it was a five-line bug in code I had added an hour earlier.

---

## 6. The one-op oracle, and why an autofix skill is safe

Each new handler was checked by tracing a one-op function on a `(1, 4, 32, 128)` bf16 input:

```
sparse_matmul (control)  exit=0  OK
getitem                  exit=0  OK
ones_like                exit=0  OK
softplus                 exit=0  OK
repeat_interleave        exit=0  OK
```

Five cases, about a minute, and it exonerated every handler when qwen's end-to-end run was still failing — which
is exactly the diagnostic value an automated fixer needs. A wrong output shape is rejected by the MLIR verifier
immediately, so the loop is **propose → trace one op → read the error**. Where a verifier only checks rank,
compare against the real `ttnn` op's output shape at the live call site.

The three mechanical tiers, in cost order, and what this audit says about each:

1. **Port from the sibling tracer.** `sparse_matmul` was the last real instance in that direction; what remains
   (`pow`, `pow_tensor`, `rearrange`) the allowlist generator covers.
2. **Generate from the allowlist.** `supported_ops.py`'s six categories each have one fixed shape rule — exactly
   what `BaseOpHandler` exploits on the TTIR path.
3. **Derive from the verifier.** `SparseMatmulOp::verify` (`TTNNOps.cpp:2720`) states the shape relation per
   sparse mode in its error branches, which is invertible. `TTNNOps.td` carries only `AnyRankedTensor`, so a
   generator must read C++ — but the C++ is explicit.

**What this audit adds to that design.** Three of the eleven blockers were *not* "missing handler for op X":

- `ttnn.copy` needed a **semantic decision** (alias, not emit) that no shape rule implies;
- `softplus` needed a **decomposition** because the dialect has no op, and the honest cost — 3 ops for 1 — is a
  judgement about what the consumer tolerates;
- `__getitem__` was not an op at all, and no op-name-driven fixer would ever have looked for it.

So a skill should be scoped to **"unblock the capture and report what it did"**, not "add op support". Its rule
should be: on failure, name the frame; if it is `_unhandled` or a TracedTensor TypeError, try the three tiers; if
it is a dunder or a semantic in-place op, stop and describe the choice for a human. And it must always report the
approximations it introduced — a decomposition or an alias changes what the advisor counts, and silently absorbing
that is how a capture starts lying.

---

## 7. Where this leaves the corpus's numbers

- **Reachable value is a lower bound for a third reason.** **17 new candidates** sit outside it — 11 on
  north-mini B (632.1 µs/model), 2 on gemma sliding (141.6 µs/model), 2 on north-mini onA full MoE (61.9 µs/model).
- **Two of seven zeros were not "no headroom" and not "coverage gap" either — they were both.** north-mini B's
  zero was honest about what it measured and wrong about what was measurable.
- **"3 coverage gaps" undercounted.** north-mini B and FN were also coverage-limited, and neither is in that count. On cause: One was a single missing handler
  (north-mini), one was a capture that substituted a dense MLP for the routed one (gemma), and one was four
  gaps of three different kinds (qwen).
- **Coverage was not the binding constraint everywhere it looked like it was.** Two of the eight fixed cell/kinds
  still cannot screen, because their harness noise floor exceeds the ceiling. That is a different defect, already
  filed as STG-5 / I5.

**Artefacts.** The tracer work is committed and pushed:
[`mvasiljevic/ttnn-jit-tracer-coverage-gaps`](https://github.com/tenstorrent/tt-mlir/tree/mvasiljevic/ttnn-jit-tracer-coverage-gaps) @ `756e134a1b`, from `618cd4e75d`.
`ttnn-jit-tracer-gap-handlers.patch` in this directory is the same diff. Reconciliations:
`advchal-v2-nm-sparse-ported-*`, `advchal-v2-nm-B-{full,sliding}-moe-*`, `advchal-v2-gemma26-{full,sliding}-moe-*`.

**What is not in the branch:** the capture-script edits that let each cell reach its own MoE tail. Those are
per-cell (gemma's dual MLP tail, north-mini B's and FN's truncated returns, qwen's token mixer) and belong with the
cells, not with `ttnn-jit`. Each is described where it is measured, above.
