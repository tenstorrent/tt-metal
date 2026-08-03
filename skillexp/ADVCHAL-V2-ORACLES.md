# advchal-v2 — correctness-oracle audit

Stage 02b lets a measured win ship only if a **correctness oracle** also passes. This file records the
oracle each cell actually built, the bar it held itself to, and the PCC it achieved.

It exists because **the oracle was not comparable across cells, and in one cell it decided the
published outcome.** Read [`ADVCHAL-V2-READ-THIS.md`](ADVCHAL-V2-READ-THIS.md) §5 for the consequence.

## Two different oracles were built under one name

| kind | reference | what it proves | what a tight bar means |
|---|---|---|---|
| **absolute** | HuggingFace / functional-decoder output for the same layer | the candidate is still a correct implementation of the model | bar ≈ 0.995–0.999 is the *model's* accuracy requirement |
| **differential** | the **frozen incumbent's** own output, same weights and inputs | the candidate changed nothing the incumbent did | bar → 1.0 demands *bit-identical* arithmetic |

The skill asks for a real-weight oracle. It does not fix which of these two you build, nor the bar.
So a cell that built a *differential* oracle and set the bar near 1.0 was asking "is this bitwise
unchanged?" — a question no re-grid of a reduction can answer yes to, because splitting a
sum-of-squares across more cores changes the floating-point summation order by construction.

## Per cell

| cell | oracle kind | bar | PCC achieved | verdict |
|---|---|---|---|---|
| phi A | differential, real weights | 0.995 | **0.9999987790** | pass → shipped −8.75 %/model |
| phi B | absolute, real HF | 0.995 | 0.998920 | pass → shipped −5.74 %/model |
| **phi FN** | **differential**, real weights | **0.999999** | **0.9999910667** (combined)<br>1.0 (RoPE only) | **combined REJECTED**<br>RoPE-only shipped −4.91 % |
| phi exp17 | absolute, real HF batch-32 | 0.995 | 0.9999834179 | pass (nothing shipped anyway) |
| gemma-4-26B exp | absolute, real HF | 0.995 | 0.998358 prefill / 0.999499 decode | pass → shipped −147.9 µs/model |
| gemma-4-12B | absolute, real HF layer-0 | 0.9868869619 | 0.999613 prefill / 0.9998707 decode | pass → shipped −1.14 %/model |
| g26 onA | absolute, real HF, 3 cases | 0.995 | 0.999629 sliding / 0.999787 full | pass → shipped −12.98 %/layer |
| g26 FN | absolute, real HF | 0.995 | > 0.995 | pass → shipped −2.04 %/model |
| nm FN | absolute, real layer-1 tensors remapped onto the layer-4 path | 0.995 | **0.999526** | pass → shipped −10.23 %/model |
| nm B | absolute, real weights | 0.9868869619 | pass | nothing shipped (all candidates slower) |
| qwen | differential, real checkpoint | — | **1.0** | pass → shipped −445.69 µs/model |
| qwen B | — | — | — | nothing shipped, no oracle needed |
| llama-3.1-8B | — | — | — | nothing shipped |
| llama-3.2-1B | — | — | — | nothing shipped |
| nm onA | — | — | — | nothing shipped (nothing measurable) |

## The inconsistency, stated plainly

**phi FN is the only cell in the corpus that used a 0.999999 bar.** Every other cell used 0.995, or a
model-specific recorded reference value (0.9869 for the two cells that had one).

Three facts follow:

1. **The same model passed a differential oracle at a lower PCC in a different arm.** phi A shipped at
   PCC 0.9999987790 — which is itself *below* 0.999999. phi A's win would have failed phi FN's bar.
2. **The bar was stricter than the model's own correctness requirement.** phi FN's shipped
   real-weight test passes at PCC **0.998902**. It rejected a candidate at 0.9999910667 — about
   120× closer to the reference than the thing it shipped.
3. **A cell that shipped a comparable change passed at 0.999526.** nm FN re-gridded a 1-core MoE
   RMSNorm to 32 cores — the same class of change, the same reduction-order effect — and its oracle
   passed at PCC 0.999526, three orders of magnitude looser than the bar phi FN failed.

phi FN's own words, from its transcript:

> The combined set is rejected despite being faster: its real-weight differential PCC moved to
> `0.9999910667` versus the frozen incumbent, while the RoPE-only candidate is bitwise-equivalent
> (`PCC 1.0`). Per the placement-stage oracle rule, I'm shipping RoPE-only and keeping the
> 11/12/24-core norm knob default-off.

The reasoning is internally consistent — it applied the strictest reading of "a placement change
should not change numerics". The defect is in the stage, not the cell: the stage never said which
oracle to build or where the bar sits, so the strictest reading was available and cost 8.3
percentage points of measured layer-level speedup.

## Fix for the stage

`final.json` should carry three required fields instead of one free-form `oracle_passed`:

```
"oracle_kind":      "absolute" | "differential",
"oracle_pcc_bar":   <number>,          # must equal the model's own test bar for `absolute`
"oracle_bar_source": "<file:line the bar was read from>"
```

and the gate should **fail** a cell that invents a bar tighter than the model's own shipped test bar
without recording a justification. A differential oracle is the right instrument for asking "did this
placement change perturb anything?", but its answer must be read as an *observation*, not a veto —
the veto belongs to the absolute oracle at the model's own bar.

## Other oracle-construction findings worth keeping

- **nm FN** had no real weights for its full-attention MoE layer. It did not fall back to the skill's
  synthetic oracle; it remapped official layer-1 tensors onto the layer-4 path transparently and
  disclosed the remap. An earlier version of the same cell had *programmatically copied* layer 1's PCC
  into the layer-4 artifact — a reviewer caught it, and it was replaced with the remapped oracle.
- **gemma-4-12B** found its local 12B cache held only the config. Rather than use the synthetic
  oracle, it selectively fetched the layer-0 shard so the shipped change had real weights behind it.
- **gemma-4-12B**'s narrow oracle *missed a real bug*: leaving K height-sharded violates the per-head
  norm contract. A broader optimized-decoder regression run caught it, the earlier measurement was
  disqualified, and the shipped form was corrected to K in **L1 interleaved**. A per-layer PCC oracle
  is not a substitute for the model's regression suite.
- **phi exp17** initially recorded its real-weight test as *skipped* because the pinned cache path was
  absent, then found the snapshot under the configured `HF_HOME` and ran a real batch-32 oracle. It
  would have closed as complete on a skipped oracle had it not re-audited itself.
- **g26 onA** and **g26 FN** both ran the oracle against the *actual shipped default* with every
  candidate environment override unset — closing the gap between "what was timed" and "what ships".
  Not every cell did this.
