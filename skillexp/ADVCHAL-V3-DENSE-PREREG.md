# Dense cells + negative control — expectations, written before the results

**Pre-registered 2026-08-11 11:10 UTC, with phi-exp17 in flight and the other two not yet started.** Writing it
down first is the whole point: in the v2→v3 build I derived predictions from the same document as the changes and
every one of them was wrong in the direction that flattered my work
([`ANALYST-PITFALLS`](ADVCHAL-V3-ANALYST-PITFALLS.md)). This file cannot be adjusted afterwards without the diff
showing it.

## The rule I am applying this time

**An expectation is floored by what the cell has already been measured to deliver from the same baseline.**
([`EXPECTATIONS`](ADVCHAL-V3-EXPECTATIONS.md) §2a — the rule that came out of publishing a 9.6 % "upper bound" for
a cell that had banked 13.0 %.) All three cells banked **exactly +0.000 %** in v2, so the floor is 0 and any
non-negative prediction is admissible. That makes these cells a weak test of the floor rule and a **strong test of
the opposite failure — shipping something that is not there.**

## What was already measured, and why these three are the calibration

| cell | v2 outcome | v2 Δ | per-layer µs | noise floor µs | flagged pool µs | signal-to-floor |
|---|---|---:|---:|---:|---:|---:|
| phi-3.5 `exp17` | `no_change` | **+0.000 %** | 1100.9 → 1100.9 | 1.092 | 83.551 | **76.5×** |
| **llama-3.1-8B `exp17`** | `contribution_zero` | **+0.000 %** | 665.0 → 665.0 | 0.697 | 4.394 | **6.3×** |
| llama-3.2-1B `exp17` | `no_change` | **+0.000 %** | 373.1 → 373.1 | 0.146 | 2.822 | 19.3× |

All three are marked **`measurable`** — the pool exceeds the floor by 6–77× — so v2 shipped nothing *despite*
having headroom to measure. That is what makes them a zero point rather than a null result.

## Predictions

### phi-3.5 `exp17` — I expect it to ship, and small

**−0.5 % to −1.5 %.** Reasoning, and it is a mechanism not an extrapolation: v3's cliff check mechanically ranks
1-core ops by profile cost, phi's dense layer has 1-core input norms, and **phi `fuse-noadvise` already proved that
knob expressible and passing at `input_norm_cores=11` for −1.08 %.** exp17 has the corpus's largest
signal-to-floor (76.5×), so if the pool is real the cliff check should find it.

Falsifiable sub-predictions:

1. **Whatever it ships will be decode-only** — every knob in this corpus is ([`CORE-ISSUE`](ADVCHAL-V3-CORE-ISSUE.md)).
2. **If it is an input-norm re-grid, its output feeds QKV → the "unsafe" class** by the rule in
   [`GUARD-FINDING`](ADVCHAL-V3-GUARD-FINDING.md) §8a. Its snapshot oracle *does* prefill —
   `_run_prefill_decode_pcc(..., seq_len=32, ...)` — so unlike north-mini it **can** see a phase inconsistency.
3. **It should nonetheless pass**, because phi is dense: no `topk` over 128 experts, so no selection boundary to
   cross, and phi measured **+3.9 × 10⁻⁶** for exactly this pattern on `fuse-noadvise`.
4. **If it fails correctness, my rule's dense exemption is wrong** and the mechanism I traced is incomplete.

### llama-3.1-8B `exp17` — the control. It must report 0.0 %

**And the specific risk is not that it finds a real win — it is that it ships a sub-band non-result.** Its flagged
pool is **4.394 µs** against a **0.697 µs** floor. A "win" of one or two microseconds is inside any honest band, and
v3's cliff check is more aggressive than v2's.

**The precedent landed this morning.** `gemma-4-12B exp11`, the dense cell that finished before the pause, shipped
`advisor_concat_l1=true` for **62,300.2 → 62,288.8 µs = −11.5 µs/model — 0.018 %** — with `oracle_pcc: None` at top
level. Eleven microseconds on a sixty-two millisecond model.

So my prediction is deliberately two-sided:

| outcome | what it would mean |
|---|---|
| **`no_change` / +0.000 %** | the control holds; every percentage in [`RESULTS`](ADVCHAL-V3-RESULTS.md) has a verified zero point |
| **ships < ~3 µs/model** | **the more likely failure, and the one to watch.** `shipped` would not mean `improved`; it would mean the cliff check manufactures candidates from a pool the size of its own noise. `RESULTS` would then need to separate *shipped* from *established* |
| ships > 1 % | the control is not a control, and something is wrong with either the incumbent or the attribution |

### llama-3.2-1B `exp17` — 0.0 %, and the tightest band in the corpus

Pool 2.822 µs, floor 0.146 µs, v2 zero. Same two-sided reading as above; its ±2.3 µs model band is the smallest
anywhere, so a sub-band ship is the easiest to detect here.

## The corpus-level question these three actually settle

Not "how much more can v3 find". It is: **does `outcome: shipped` in this corpus mean anything?** gemma-12b says
maybe not. If two of the three dense cells ship µs-scale changes, then the honest form of the results table has
three columns — *shipped*, *outside its own band*, *and reproduced* — and the −6,769 µs total needs restating on
the middle one.

## What I am not predicting, and why

I am **not** predicting a µs/model total for the three. The capacity metric that produced the last corpus-level
number was refuted by its own inputs, and I have not built a replacement. An honest "I don't know, here is the
floor and here is the failure mode to watch" is worth more than a number I would have to retract.


---

# RESULT — phi-3.5 `exp17`, scored against the prediction above

`rc=0`, gate PASSED with no warnings, published as `skillexp/done/advchal-v3/exp17/microsoft_phi_3_5_mini_instruct`.
Driver then held at the re-placed sentinel before the llama control, as intended.

| | |
|---|---|
| outcome | **`improved`** |
| model estimate | 32,407.3 → 31,339.4 µs = **−1,067.9 µs = −3.295 %** |
| band | 61.7 µs → **17.3× the band** — solidly established |
| oracle | absolute, **PCC 0.9999810562728417**, bar 0.995 |
| **incumbent PCC** | **0.9999810562728417 — identical to the candidate's** |
| shipped | **`{"advisor_rope_l1": true, "advisor_norm_core_count": null}`** |

## Scoring my prediction

| prediction | outcome |
|---|:--|
| "it will ship" | ✅ |
| **"−0.5 % to −1.5 %"** | ❌ **wrong — −3.295 %, 2.2× above my upper bound.** Underpredicted, same direction as the corpus-level miss |
| "most likely an input-norm re-grid" | ❌ **wrong** — it shipped the **rope** knob; `advisor_norm_core_count` is `null`, no norm re-grid at all |
| "whatever it ships will be decode-only" | ✅ — the rope knob lives in the decode path |
| "should pass because phi is dense" | ✅ **and stronger than predicted** — not merely passing but **bit-identical PCC to the incumbent** |
| "if it fails correctness, my dense exemption is wrong" | the exemption **survives** |

Two of four, and the quantitative one wrong by 2.2× in the direction that flatters v3.

# ⚠ And the decisive finding: the same knob, the same stage, two agents, one correct

**phi-exp17 shipped `advisor_rope_l1: true` for −3.295 % at a PCC identical to the incumbent's.
phi `nofuse-noadvise` wrote the same knob and scored 0.9849539 (my re-measure) / 0.9173130 (its own record) and was
rejected.**

| | phi `nofuse-noadvise` | **phi `exp17`** |
|---|---|---|
| knob | `advisor_rope_l1` | **`advisor_rope_l1`** |
| stage | frozen v3, `4ea2fb1fb7d` | **the same, unmodified** |
| skill, prompt, gate | identical | identical |
| PCC | **0.9849539 — fails** | **0.9999810562728417 — identical to incumbent** |
| shipped | **nothing** | **−1,067.9 µs** |

Same model, same knob, same stage, same guidance. **One agent wrote a correct implementation and one wrote a
defective one.** So:

1. **The stage is capable of producing this change correctly** — it just did, unaided, on the unmodified tree. My
   claim in [`GUARD-FINDING`](ADVCHAL-V3-GUARD-FINDING.md) §7b that the remedy is *"a known-good replacement"* is
   true but understates the options: **a re-run may simply get it right.**
2. **Part of the v3 regression is agent implementation variance on hand-written model code**, not stage design. That
   is the hypothesis I raised in [`WHY-WORSE`](ADVCHAL-V3-WHY-WORSE.md) §3 and then set aside as unfalsifiable
   because I had no matched pair. **This is the matched pair.**
3. **And the stage cannot tell the two apart.** Gating on the layer PCC conflates *"my placement idea is bad"* with
   *"my code is wrong"*. phiB's agent concluded the former and shipped nothing; the correct conclusion was the
   latter. That is the same conflation as gemma-onA's, at a different level.

# The band audit this prompted, across all 13 cells with results

| | µs |
|---|---:|
| total shipped, 13 cells | **7,849.0** |
| **outside its own band** | **7,837.6 — 99.85 %** |
| **inside its own band** | **11.5 — gemma-4-12B `exp11`, at 0.03× its band** |

Ratios: phiA **41×**, nmOnA **30×**, phi-exp17 **17×**, g26FN **16×**, phiFN **9×**, nmFN **8×**, g26onA **3.6×**,
qwenB **3.0×**, nmB **1.5×** — and **gemma-4-12B 0.034×**, i.e. **29× inside its own uncertainty band, reported as
`outcome: shipped`.**

**So the "shipped ≠ established" worry I pre-registered lands on exactly one cell, and it is not the control.**
Every other shipped result clears its own band, most of them comfortably. The corpus total should be quoted as
**−7,837.6 µs established**, with gemma-12b's 11.5 µs recorded as a sub-band non-result rather than a win.
