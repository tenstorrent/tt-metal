# What the agent actually got wrong — why the PCC was bad, and how it could have known

Not *"how it could have been more careful"*. This is the specific reasoning error, the specific signal that was
sitting in front of it, and the specific cheap checks that would have converted a rejection into a fix.

**The one-line answer: the bad PCC was never a correctness failure of the change the agent made. It was a discrete
routing decision flipping because a continuous quantity moved by less than one bit — and the agent had, in the
numbers already on its screen, a 1000× magnitude inconsistency that said so.**

---

# 1. Why the PCC is bad

Measured end to end ([`GUARD-FINDING`](ADVCHAL-V3-GUARD-FINDING.md) §6–7):

| step | what happens | measured |
|---|---|---|
| 1 | the agent width-shards the residual RMS-norms **in decode only** | — |
| 2 | the sharded and interleaved reductions differ — **but neither is wrong** | 1 bf16 ULP in 18–36 % of channels; **both within 1.4 × 10⁻⁶ of float64** |
| 3 | site 1 is `input_ln` → QKV → **the KV cache write**. Prefill filled the cache interleaved; decode appends sharded | — |
| 4 | attention over that hybrid cache perturbs the residual, and so the **router logits** | max \|Δlogit\| = **0.046875** |
| 5 | the gap between the **8th and 9th** expert logit | **0.015625** |
| 6 | **the top-8-of-128 selection flips** | expert 104 → 123 |
| 7 | a different expert set is a different function | layer PCC **0.99963 → 0.99437** |

**Proof that step 6 is the whole story, not a candidate:** pinning the routing vector to the incumbent's expert set
while leaving *every* sharded placement in force recovers **99.4 %** of the loss (0.9995948 against a 0.9996280
baseline). Conversely, feeding the sharded routing into an otherwise interleaved layer reproduces **100 %** of it.

So: **nothing the agent changed computes a wrong answer.** The reduction it moved is accurate to 10⁻⁶. The
5.2 × 10⁻³ is a *different function being evaluated*, because a tie-break moved.

---

# 2. What the agent got wrong — four errors, in order of how cheaply each was avoidable

## 2.1 It read a diagnostic as a verdict — this is the real error

The agent had four numbers:

| | |
|---|---|
| the bar | 0.995 |
| the incumbent's layer PCC | 0.99963 |
| its candidate's layer PCC | 0.99457 |
| what it changed | **the memory placement of a reduction — not its arithmetic** |

From those, it concluded: *"the sliding kind's residual-norm re-grid is incorrect; reject the kind."*

**The conclusion available from the same four numbers was the opposite:** a placement change that preserves the
arithmetic cannot legitimately cost 5 × 10⁻³ at layer scope. A reassociated bf16 reduction moves individual
elements by ~1 ULP; expressed as a PCC over 2816 channels that is **order 10⁻⁶**. The agent measured **5 × 10⁻³ —
about a thousand times larger than the largest effect its own change is capable of.**

> **A measurement three orders of magnitude larger than the physics of your change is not a verdict on your change.
> It is evidence of a coupling you have not found.**

That inference needed **no device time and no new tooling**. It is arithmetic on numbers already in the artefact.
And it is the difference between "reject, −5,919 µs written off" and "investigate, one policy field, win kept".

## 2.2 It changed one phase of a two-phase computation without asking what the other phase assumed

`_rms_norm` *is* a pure function of its input — the agent's mental model was not wrong about the op. What it missed
is that this op's **output is persisted by one phase and consumed by another**: site 1 feeds the QKV projection,
whose K/V go into the KV cache during prefill and are read during decode. Sharding decode alone makes the cached
entries and the new entry products of different reductions.

Nothing in the agent's environment made that visible: the harness measures **traced decode replay**, the gate is
**one PCC number**, and the incumbent it inherited **already gated a different knob on `_executing_decode`** (a
DRAM-sharded dense matmul), so the local idiom said "decode-gating is how we do this here". **The agent followed
the convention of the code it was handed.** That is a defect of the *surrounding* system as much as of the agent.

## 2.3 It generalised one measurement onto sixteen untested configurations

Having decided the kind failed, it applied that verdict to all 17 sliding measurements — 16 of which carry
`"oracle_pcc": "not_retested; sliding kind vetoed at its best measured rung"`. As it happens **all of them do fail**
in that tree, so the outcome was right; but the procedure could not have known that, and its own `README` records
the shortcut in plain words: *"only the fastest rung was spent on the absolute oracle."*

## 2.4 It never tried the advised value, and wrote the verdict as a literal

The ladder ran 2/4/8/11/22/44. **88 — the advisor's actual advice, and the only rung that passes under a consistent
guard — was never measured.** And `build_evidence.py` records `passed = kind == "full_attention"` with the PCC
hardcoded beside it and no oracle log committed, so nothing downstream could re-derive any of it.

---

# 3. How the agent could have known what the number meant

Five checks. The first costs nothing, and the most expensive costs four minutes.

## 3.1 Compare the deviation against what your change can physically do — **0 seconds**

*"I moved a reduction. Reductions reassociate. In bf16 that is ~1 ULP per element, ~10⁻⁶ on a 2816-wide PCC. I
measured 5 × 10⁻³. That is 1000× too large, so the norm is not the thing failing."* Pure arithmetic on numbers
already in hand. **This alone flips the verdict from "reject" to "investigate".**

## 3.2 Measure the op alone — **30 seconds**

Same shape, same shard spec, same compute config, sharded vs interleaved, PCC against float64. Returns
**0.9999986** and settles it. The agent built a per-candidate harness for the *layer*; it never once measured the
*op*.

## 3.3 Ablate against the oracle, not just against the clock — **4 minutes**

**The agent already had the tool.** It ran **seven single-site ablations** — and used every one of them for
*timing* only. Running those same seven against the oracle localises the entire 5.2 × 10⁻³ to **site 1** and hands
you the fix (`drop_index=1`, PCC 0.9996227, −11.47 %/layer). The instrument existed and was pointed at the wrong
question.

## 3.4 Vary the fixture the gate holds fixed — **2 minutes**

The oracle hardcodes `seq_len = 32`. The same candidate at seq 8 scores **0.9991 and passes**; at seq 4,
**0.9992**. **A candidate whose correctness depends on the prefill length is not an incorrect candidate — it is a
cross-phase inconsistency, announcing itself.**

## 3.5 Read the other phase — **1 command**

`grep -n rms_norm` in `prefill_forward` / the prefill branch. The asymmetry is one line of output. For north-mini
the same grep shows `ttnn.rms_norm(hidden_states, eps, weight)` in prefill against
`memory_config=self.decode_norm_memory_config` in decode.

---

# 4. How it could have fixed it — all three measured

| fix | PCC | Δ/layer | robust across prefill lengths? | intrusiveness |
|---|---:|---:|:--|---|
| **`drop_index=1`** — leave the cache-visible site interleaved | **0.9996227** | **−11.47 %** | **yes** — tracks the baseline at seq 8/32/64 | one policy field |
| phase-consistent guard — shard prefill too | 0.9996293 | −12.27 % | **no** — only fires when prefill ≤ 32 rows | one line, but wrong in production |
| stabilise the routing | 0.9995948 | (all sharding retained) | untested | addresses the real fragility |

**The best available answer was more accurate than the incumbent *and* 11.5 % faster.** There was never a
correctness/latency trade here; the trade appeared only because every configuration the agent measured had the
inconsistency baked in.

---

# 5. The frame error underneath all of it

The agent's model of its job was: **propose a placement → measure it → gate it on a PCC bar.** In that frame, a PCC
under the bar means the placement is bad, and rejection is the correct action. The frame is what failed.

**The bar is a whole-layer integrity check being used as a per-op correctness gate, and on this model those are
different instruments.** A whole-layer PCC on a sparse-MoE decoder is dominated by *discrete expert agreement*: a
1-ULP perturbation flips a selection on ~1 % of tokens regardless of what caused it, so the metric has a
discontinuous floor unrelated to the arithmetic of the change under test. Judging a norm's placement by it is using
a smoke alarm to read temperature — it fires, and the reading says almost nothing about the thing you touched.

**What a correct frame looks like, and it is not more work:**

1. **Gate the op on the op.** A placement change is correct iff the op's own output is unchanged to within the
   reassociation bound. That is a 30-second isolated test and it is a *stronger* check than the layer PCC.
2. **Keep the layer PCC as an integrity check, and treat a breach as a question.** *"What did I perturb, and what
   downstream decision is sensitive to it?"* — not *"my change is wrong."*
3. **Declare cross-phase behaviour for anything whose output is persisted.** If the op feeds the KV cache, its
   prefill and decode paths must agree, and that is an assertion, not a hope.
4. **Never generalise one oracle sample.** It binds the configuration it was measured on.

---

# 6. And I made the same error, in the same direction

Worth stating, because it is evidence about the error rather than about the agent. Handed the same 0.9946, I
published a **candidate tt-metal kernel bug** — `rms_norm` allegedly non-monotonic in core count, "three
independent reproductions". The magnitude argument in §3.1 rules a kernel bug out in one line: a norm accurate to
10⁻⁶ cannot produce 5 × 10⁻³. **I then spent 79 device configurations learning what that one paragraph would have
told me**, and the three reproductions turned out to be one policy measured three times.

So the failure is not that the agent lacked care or tooling. **Both of us reached for the instrument that was
already pointed at the layer, and neither of us asked whether the number it produced was even capable of describing
the change we had made.** That question — *can my change physically do this?* — is the one durable lesson here, and
it costs nothing to ask.
