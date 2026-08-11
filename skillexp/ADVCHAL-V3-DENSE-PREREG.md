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
