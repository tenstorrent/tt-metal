# The core issue: the harness chose the shape of the code, and each model's fixture chose which defects were visible

**Read [`WHAT-THE-AGENT-GOT-WRONG`](ADVCHAL-V3-WHAT-THE-AGENT-GOT-WRONG.md) alongside this.** That file answers the
narrower and more useful question — *why is the PCC bad, how could the agent have known what the number meant, and
how could it have fixed it* — and its answer is that **a placement change that preserves arithmetic to 10⁻⁶ cannot
cost 5 × 10⁻³ at layer scope, so the 1000× discrepancy was itself the signal.** This file is the systemic account;
that one is the agent's actual reasoning error.

**v3 did not underperform v2 because it searched worse or judged worse. It underperformed because the one thing it
did not change — the *scope* of its measurement — is the thing that determined both what its agents wrote and what
its gate could see.** Everything v3 added (absolute oracle, legal ladder, cliff check, fresh-process confirmation,
provenance) operates *inside* that scope, and this defect class lives outside it.

Established by measurement in [`GUARD-FINDING`](ADVCHAL-V3-GUARD-FINDING.md); this file is the audit across all
three models and what it means.

## The chain, in four links

**1. The harness measures traced decode replay of one layer.** Every `measurements/*.json` in the corpus carries
`harness_scope: "one … decoder layer, traced decode replay, batch N, measured end to end on host"`. Decode is the
unit of work, so decode is where a knob has to act to be measurable.

⚠ **Sharpened by the qwen audit — see [`GUARD-FINDING`](ADVCHAL-V3-GUARD-FINDING.md) §8a.** Decode-only gating is
**not** the defect. qwen's shipped knob is decode-only too and is **bit-identical on both layer kinds** with a real
−1.0 %, because it shards the **MLP**, whose output never outlives the step. The precise condition is:
**a decode-only knob is unsafe iff its output flows into the KV cache write.** That rule classifies all seven
audited knobs correctly and is a static question, answerable before any measurement. Read link 2 below with that
correction in mind.

**2. So every agent wrote a knob that only takes effect in decode** — four models, four different spellings of the
same thing:

| model | knob | how it is gated | effectively decode-only? |
|---|---|---|:--|
| gemma-4-26B (v3) | `advisor_residual_norm_cores_by_kind` | `self._executing_decode` | **yes**, explicitly |
| north-mini (v3) | **`decode_norm_cores`** | only referenced inside `decode_forward` | **yes**, structurally |
| phi-3.5 (v3) | `input_norm_cores` | `tuple(shape) != (1,1,batch,hidden) → super()` | **yes** — prefill shape never matches |
| phi-3.5 (v2 and v3) | `advisor_rope_l1{_chain}` | inside `_decode_rope` | **yes**, both versions |
| gemma-4-26B (**v2**) | `advisor_norm_cores` | `x.shape[-2] > TILE_SIZE → skip` | **no — shards prefill too, if prefill ≤ 32 rows** |

The last row is the whole difference between the versions on the cell that decided the corpus. **v2 wrote a
shape-based guard and got prefill/decode consistency by accident; v3 wrote a phase-based guard and lost it.**

**3. Those knobs all change the arithmetic that produces K/V** — the only state that crosses from prefill into
decode. north-mini's `decode_norm_cores` shards the norm feeding `_attention_decode`; phi's `input_norm_cores`
shards the norm feeding QKV; phi's rope knob changes K directly; gemma's site 1 is `input_ln`. Sharding one phase
and not the other means the cached entries and the newly-computed ones come from different reductions.
[`GUARD-FINDING`](ADVCHAL-V3-GUARD-FINDING.md) §3 shows this is the *only* site of the eight that matters: leaving
site 1 interleaved recovers the whole loss while the other seven stay sharded.

**4. Whether the gate can see the resulting inconsistency is decided by one constant in each model's oracle
fixture** — and those constants were chosen by whoever wrote the model, for unrelated reasons:

| model | oracle's prefill before the decode step | can it see a prefill/decode inconsistency? |
|---|---|:--|
| gemma-4-26B | `seq_len = 32` → prefill(32) → decode | **yes** — measured cost 5.3 × 10⁻³ |
| phi-3.5 | `prefix_length = 127`, batch 32 | **yes** — but the model measures insensitive to it (+3.9 × 10⁻⁶) |
| **north-mini** | **`create_paged_kv_cache()` then `decode_forward` at position 0** | **NO — empty cache, no prefill at all** |

north-mini's oracle:

```python
key_cache, value_cache = decoder.create_paged_kv_cache()          # fresh, empty
current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0])   # position 0
actual = decoder.decode_forward(hidden_tt, key_cache=key_cache, ...)
_assert_pcc("optimized-real-layer1-decode", reference, …, threshold=0.995)
```

**There is no prefill.** The cache holds only the entry the decode step writes itself, so prefill/decode agreement
is not a property the test has. And the gemma prefill-length sweep says exactly what that means: the cost of the
inconsistency was **9.4 × 10⁻⁵ at 4 cached tokens, 1.2 × 10⁻⁴ at 8, 5.3 × 10⁻³ at 32**. At **0** it is zero by
construction.

## What that produced, cell by cell

| cell | shipped | touches K/V? | oracle prefill | oracle margin | µs/model | status |
|---|---|:--|---:|---:|---:|---|
| gemma-4-26B `-onA` **sliding** | **nothing — vetoed** | yes | 32 | **breach, −5.2 × 10⁻³** | **0** *(−5,919 lost)* | the pattern was **expensive and visible** |
| gemma-4-26B `-onA` **full** | 7 norms @ 8 cores | yes | 32 | 4.80 × 10⁻³ | −1,198 | pattern present, costs 7 × 10⁻⁶ — **passes by luck of kind** |
| **north-mini `-onA`** | **`decode_norm_cores=16`** | **yes** | **0 — empty cache** | **9.78 × 10⁻⁴** | **−1,400** | **pattern present and untestable by its own gate** |
| north-mini `fuse-noadvise` | `advisor_sliding_norm_cores=22` | yes | — | 4.52 × 10⁻³ | −351 | pattern present |
| phi-3.5 `fuse-noadvise` | `input_norm_cores=11` | yes | 127 | ~3.9 × 10⁻³ | −278 | pattern present, model insensitive |
| north-mini `nofuse-noadvise` | `decode_topk_cores=110` | **no** — within-step | — | 3.27 × 10⁻³ | −171 | not affected |
| gemma-4-26B `fuse-noadvise` | concat→projection boundary | no | 32 | — | −986 | not affected |
| phi-3.5 `-onA` | rope/L1 chain | yes (rope → K) | 127 | 4.99 × 10⁻³ | −1,254 | pattern present in **both** versions |
| qwen3.6 `nofuse-noadvise` | `advisor_plan=mlp_product_only` | **no — MLP, within-step** | b32 both kinds | — | −1,130 | ✅ **audited: bit-identical PCC both kinds, clean** |

**−3,227 µs of v3's −6,769 — 48 % — is shipped on knobs that touch a K/V producer** (qwen's −1,130 is now audited and clean, north-mini's −1,400 is measured safe), and the single largest piece of it
(**−1,400 µs, north-mini `-onA`**) was gated by an oracle that cannot detect the defect at all, passing by
**9.8 × 10⁻⁴** against a mechanism that cost gemma **5.2 × 10⁻³**.

⚠ **This is not a claim that those wins are wrong.** The cost of the inconsistency varies by more than **700×**
across models and layer kinds — 7 × 10⁻⁶ on gemma `full`, 3.9 × 10⁻⁶ on phi, 5.2 × 10⁻³ on gemma `sliding`. It is
a claim that **nothing in the stage predicts which**, and that for north-mini nothing measured it either.

## Why the cost varies so much, and where gemma's sensitivity comes from

The one kind where it is expensive is gemma's **sparse-MoE sliding** kind, and the earlier router experiment
explains it: `ttnn.topk(k=8)` over **128 experts** flips a selection on ~1 % of tokens under a 1-ULP perturbation.
A 1-ULP change in the cached K propagates into the attention output, into the residual, into the router logits —
and a routing flip is a **discontinuity**, not a rounding. So the same 1-ULP cause is worth 10⁻⁶ on a dense layer
and 10⁻³ on a sparse-MoE one.

That also means **the sensitive cells are exactly the cells v3 unlocked.** The three "coverage win" cells —
40 % of v3's total — are the sparse-MoE kinds that v2's tracer never captured. v3 made them visible, searched them
harder, and shipped placements into the one regime where this defect class is expensive.

## So: why did v3 underperform v2?

Not a search failure and not a judgement failure. Three things compounded:

1. **v3 wrote phase-gated knobs because its harness measures a phase.** v2's shape-gated guard was consistent at
   the oracle's prefill length by accident. Neither version reasoned about it; v2 got the coin-flip right.
2. **v3's gate could not attribute the consequence.** On gemma sliding the oracle *did* detect it — 0.99457 against
   a 0.995 bar — but with `oracle_passed` hardcoded to the layer kind, no oracle log, and `op_under_test` advisory,
   the cell had nothing to work with except "the norm re-grid fails". So a **model-code defect was recorded as a
   placement rejection**, and −5,919 µs was written off as unavailable.
3. **v3 searched harder in exactly the blind region.** The ladder and cliff check are real improvements; they
   increased the number of K/V-touching placements tried, on the sparse-MoE kinds that are most sensitive to them,
   under a gate whose fixture varies per model between "sees it" and "cannot see it".

**The single sentence:** *v3 improved the rigour of its measurement without changing its scope, and the defect
class it needed to see is created by that scope and hidden by it.*

## And the corollary about v2's numbers

v2 is not safe on this axis either, it is only lucky. Its guard fires only when prefill ≤ 32 rows — **measured**:
at seq 64 v2's `phase=both` returns the decode-only number to sixteen digits. So v2's gemma win passes its oracle
at exactly the one prefill length where its guard is consistent. **Neither corpus has established this class of
correctness at production prefill lengths, and neither oracle tests more than one.**

## Actions, in priority order

1. **Re-oracle the four affected shipped wins at a non-trivial prefill length**, north-mini `-onA` first: it is
   −1,400 µs on a 9.8 × 10⁻⁴ margin, gated by an empty cache. This is the only item that could change a shipped
   result rather than recover a lost one.
2. **Every model's oracle must run prefill → decode at ≥ 2 prefill lengths.** north-mini's runs none. Every
   correction in this corpus traces to a fixture constant nobody varied
   ([`PITFALLS`](ADVCHAL-V3-ANALYST-PITFALLS.md) ERROR 18).
3. **Ship gemma sliding at 88 cores with `drop_index=1`** — −5,260 µs/model, PCC 0.9996227, holds at every prefill
   length tested. [`GUARD-FINDING`](ADVCHAL-V3-GUARD-FINDING.md).
4. **A placement knob that changes a K/V producer must declare its cross-phase behaviour**, and the gate must
   assert it. This is the general rule; items 1–3 are its instances.
5. **`oracle_passed` computed from a parsed, provenanced oracle artefact.** Without it, item 2's finding would
   again be recorded as a placement rejection rather than a model defect.
6. **Add a *scope* review to the stage, not just a rigour review.** Before the next run, one question per gate
   check: *what does this check hold fixed that the deployed model varies?* Layer count, batch, prefill length,
   execution phase, cache state. Every defect in this corpus is in that list.


---

# ⚠ Correction: north-mini's win is probably safe, and I overstated the risk

The table above called north-mini `-onA` *"pattern present and untestable by its own gate"* and set it beside a
mechanism that cost gemma 5.2 × 10⁻³. **Tested, and the risk was overstated.**

Ran the model's own prefill → decode path with real layer-1 weights and compared knob-off against
`decode_norm_cores=16` as a function of prefill depth — a differential test, which is enough to detect a
prefill/decode inconsistency without rewriting its reference:

| prefill tokens | PCC(norm_off, norm_16) | max \|Δ\| |
|---:|---:|---:|
| 0 | 0.9998240581 | 9.77 × 10⁻³ |
| 4 | 0.9998727673 | 5.13 × 10⁻³ |
| 8 | 0.9998737445 | 4.40 × 10⁻³ |
| 16 | 0.9999178386 | 3.91 × 10⁻³ |
| **32** | **0.9999537598** | 3.42 × 10⁻³ |

**The divergence shrinks with cache depth — the opposite of gemma, where it grew 9.4 × 10⁻⁵ → 5.3 × 10⁻³.** So
no routing discontinuity is being crossed here: north-mini scores its experts with a **sigmoid**, not a softmax
over a top-8 of 128, and its margins are not being flipped. And its oracle's degenerate fixture (prefill 0) is the
configuration where the knob's effect is **largest** (1.76 × 10⁻⁴), not smallest — so the 9.78 × 10⁻⁴ margin it
passed by was measured at the worst case, and the effect only improves with depth.

**Conclusion: the −1,400 µs stands.** What remains true is narrower: its oracle's reference is a closed form valid
only at position 0 (`attention == V`, no Q, no K, no softmax), so **the knob's effect on the Q/K path is not
checked by its own gate** — the reassurance above comes from this differential test, not from the oracle. The
oracle should still be extended; the win should not be doubted.

## And a real defect found on the way: the shipped win cannot be turned off

`from_state_dict` contains

```python
if candidate == "default" and batch == 1:
    # Advisor-challenger winner: the 16-core rung beat both sides of
    # the advisor's 22-core choice in fresh traced-replay processes.
    policy = replace(policy, decode_norm_cores=16)
```

**It overrides the policy unconditionally.** `dataclasses.replace(POLICIES["default"], decode_norm_cores=None)`
comes back as **16**, with an identical shard spec. So:

- the shipped win is **not ablatable through the model's own policy surface**;
- **any A/B measurement of it via `POLICIES` silently compares 16 against 16** — my first attempt did exactly
  that and returned *bit-identical at every prefill depth*, which reads as "the knob has no effect";
- reproducing or re-measuring the −1,400 µs requires editing the constructor.

**A shipped optimisation that its own policy surface cannot disable is not measurable, and the failure mode is
silent agreement rather than an error.** Every other cell in this corpus expresses its winner as a policy field.
This one should too.
