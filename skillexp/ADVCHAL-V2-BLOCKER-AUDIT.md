# Every blocker this corpus recorded: which are real, which are fixable, and what fixing them bought

Seven of fifteen cells returned zero, and the docs attribute three of those to "coverage gaps" — layers the
advisor never saw. This audits every op those cells named as a blocker, decides what class of fix each needs, and
where a fix was cheap, applies it and re-measures.

**Headline:** of the ops named across the corpus, **one did not exist**, **two more blockers were never named at
all**, and the tracer-side gaps were all closed in one afternoon of pure-Python edits — no rebuild. Coverage on
four cell/kind pairs went from 58–77 % untraced to 4–15 %.

Method: re-capture each cell with the handler added, then re-run `reconcile.py` against **the cell's own profile
window and incumbent**, with the shipped report re-run through the identical command as a verified control. Where
the control did not reproduce the cell's published numbers, the comparison is not shown.

---

## 1. The verdict per blocker

| op, as the docs name it | real blocker? | what a fix needs | status |
|---|---|---|---|
| `ttnn.sparse_matmul` | **yes** | port the handler that already exists in the sibling tracer | **done, measured** — 4 cell/kinds |
| `ttnn.copy` | **yes** | tracer bookkeeping only; no dialect op exists or is needed | **done** — traces |
| `ttnn.softplus` | **yes** | no standalone dialect op; 3-op decomposition, or add `TTNN_SoftplusOp` | **done** (decomposed) |
| `ttnn.recurrent_state_update` | **NO — the op does not exist** | — | see §3 |
| `ttnn.paged_fused_update_cache` | yes | it is two `paged_update_cache` calls; both dialect ops and handlers exist | not applied |
| `ttnn.ones_like` | yes | `ttnn.ones` at the input's shape — a copy of `_zeros_like_handler` | **done**, oracle-tested |
| `pow` / `pow_tensor` | yes | `TTNN_PowTensorOp` / `TTNN_PowScalarOp` exist | not applied |
| `rearrange` | yes | no dialect op; decomposes to `permute` + `reshape` | not applied |
| `rotary_embedding_hf` *(missing from the interception tracer)* | yes | port from the emit tracer, or drop the `--help` claim | not applied |
| sharded GQA SDPA output | yes | a tt-metal kernel — out of scope | recommendation only |
| **`TracedTensor.__getitem__`** — *not in any doc* | **yes** | not an op at all: a proxy-protocol gap | **done** — traces |
| **`ttnn.repeat_interleave`** — *not in any doc* | **yes** | `TTNN_RepeatInterleaveOp` exists | **done**, oracle-tested |

Nothing here is a tt-mlir optimizer defect. Every item is either a `ttnn-jit` tracer gap, a missing dialect op, or
a tt-metal kernel limitation — and the tracer gaps are the overwhelming majority.

---

## 2. What the fixes bought, measured

Each row is the same cell, the same profile window, the same advisor pin (`618cd4e75d`), with the shipped report
re-reconciled through the identical command as a control.

| cell / kind | untraced, shipped → fixed | screening ceiling vs noise floor | chains resolvable alone |
|---|---|---|---|
| north-mini onA, full sparse MoE | **77.15 % → 14.39 %** | 0.66× → **10.09×** | **0 → 2** |
| north-mini onA, sliding sparse MoE | **76.62 % → 15.47 %** | 0.12× → 0.67× | 0 → 0 |
| gemma-4-26B onA, full attention | **58.51 % → 3.86 %** | 1.70× → 3.29× | 0 → 0 |
| gemma-4-26B onA, sliding attention | **64.70 % → 4.17 %** | 4.08× → **13.74×** | **2 → 4** |
| qwen3.6-27B B, linear attention | trace aborted → **71 ops captured** | *see §4* | *see §4* |

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

## 3. `ttnn.recurrent_state_update` does not exist

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

## 4. qwen: the coverage gap is closed, and the failure moved

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

**The failure that remains is not a coverage gap.** `ttnn-advise capture` now aborts natively inside
`mlir::PassManager::run` while placing this graph, with no diagnostic. What is established:

- each new handler passes a one-op trace in isolation (§5), so none of them emits invalid IR on its own;
- the traced module verifies and parses — `ttmlir-opt --ttnn-to-ttnn-l1-advisor` accepts it and exits 0;
- the crash is inside the pass pipeline, in-process, where the op-model constraint queries run.

**That is where the evidence stops.** The stack carries only `OpToOpPassAdaptor` frames, so attributing it to a
specific pass needs a debug build. It is *not* established that this is an optimizer defect — a tracer that emits
verifiable-but-unusual IR (a 6-D tensor, a 1536-wide batch, the three-op softplus) could equally be provoking it.
The one thing that is now certain is that **`ttnn-jit` coverage is no longer what blocks qwen**, so the ≈62 %
figure should be read as "reachable, pending one pipeline crash" rather than "unreachable".

---

## 5. The one-op oracle, and why an autofix skill is safe

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

## 6. Where this leaves the corpus's numbers

- **Reachable value is a lower bound for a third reason.** Four new candidates worth ~200 µs/model sit outside it.
- **"3 coverage gaps" was right about the count and wrong about the cause.** One was a single missing handler
  (north-mini), one was a capture that substituted a dense MLP for the routed one (gemma), and one was four
  gaps of three different kinds (qwen).
- **Coverage was not the binding constraint everywhere it looked like it was.** Two of the four fixed cell/kinds
  still cannot screen, because their harness noise floor exceeds the ceiling. That is a different defect, already
  filed as STG-5 / I5.

**Artefacts:** `ttnn-jit-tracer-gap-handlers.patch` (both tracers, 176 insertions, no rebuild),
`advchal-v2-nm-sparse-ported-{report,reconciliation}.json`. The tracer changes are **not** committed to tt-mlir —
they are working-tree edits plus a live copy in the container venv, with `.orig-618cd4e` backups beside each.
