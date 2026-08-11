# Where the 0.99963 → 0.99457 came from — isolated, on device, with real weights

**Answer: nowhere that can be reproduced.** Four candidate mechanisms were tested and all four are ruled out or
too small by orders of magnitude. Then reading the cell's own `build_evidence.py` shows why: **the sliding
kind's `oracle_pcc` is a hardcoded literal and its `oracle_passed` is the expression
`kind == "full_attention"`.** The verdict that cost 6,055 µs/model was not computed from a measurement in any
artefact this run committed.

Run 2026-08-11, device idle, both STOP sentinels in place. Real layer-0 weights fetched from HuggingFace by
**safetensors byte-range request — 721 KiB, not a 50 GiB checkout**: `router.proj.weight` [128, 2816],
`router.scale` [2816], `router.per_expert_scale` [128], `pre_/post_feedforward_layernorm_2.weight`. Scripts and
raw output in `run-config/`.

## The four candidates, and what each measured

### ❌ 1. The core count. 11 cores and 88 cores are **bit-identical**.

Element-wise, not PCC. Across three input scales, six input/weight combinations and nine grids:

| input | sharded vs interleaved | **grid vs grid (2 … 88)** |
|---|---|---|
| unit scale | **bit-identical** | bit-identical |
| ×120 scale | 477–532 of 2816 elems differ by **1 bf16 ULP** (max rel 7.75 × 10⁻³) | **bit-identical — every grid produces the same output** |
| spiked ×3000 | 295–344 elems differ by 1 ULP | bit-identical *(one exception: `4×2` differs from `8×1`, 772 vs 344 elems)* |

**So v2's 88 cores and v3's 11 cores compute the same bits.** A whole-layer PCC of 0.99963 for one and 0.99457
for the other is not a fact about the grid. This is the measurement that makes everything below necessary.

### ❌ 2. The code difference between v2 and v3. Also bit-identical.

The two trees implement the same knob differently: **v2 passes the norm weight through unchanged (DRAM
interleaved) and shards only the activation; v3 reshards the weight** into the width-sharded config. That was the
leading hypothesis after [`OP-BY-OP`](ADVCHAL-V3-OP-BY-OP-VS-V2.md) §2 found exactly this class of divergence on
phiB. Tested with the **real** `pre_feedforward_layernorm_2.weight`:

| scale | grid | v2-style vs v3-style | v2-style vs interleaved | v3-style vs interleaved |
|---:|---:|---:|---:|---:|
| 1 | 11, 88 | **0 elems** | 0 | 0 |
| 30 | 11, 88 | **0 elems** | 450 | 450 |
| 120 | 11, 88 | **0 elems** | 532 | 532 |

**Weight placement changes nothing.** Unlike phiB, the two gemma implementations are numerically the same change.

### ❌ 3. Expert-routing flips. Real router weights, and they are rare.

The hypothesis from [`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md) §3: the router norm is the `weight=None`
`rms_norm` whose output feeds `ttnn.topk(k=8)` over **128 experts**, so a 1-ULP perturbation near a routing
boundary flips a selection — a discontinuity, which would explain a step rather than a slide. Measured with the
real `router.proj.weight` and `router.scale`, 24 trials per scale, interleaved norm vs 11-core sharded norm:

| activation scale | norm elems differing | mean logit gap, 8th − 9th | experts flipped / 8 | trials with any flip |
|---:|---:|---:|---:|---:|
| 1 | 264 | 2.17 | 0.000 | 0/24 |
| 5 | 334 | 1.67 | 0.000 | 0/24 |
| 15 | 214 | 1.59 | 0.042 | **1/24** |
| 30 | 351 | 1.87 | 0.000 | 0/24 |
| 60 | 314 | 2.04 | 0.042 | **1/24** |
| 120 | 374 | 1.79 | 0.000 | 0/24 |
| 240 | 339 | 1.73 | 0.000 | 0/24 |

**2 flips in 168 trials — 1.2 % of tokens, and never more than one slot of eight.** The 8th-to-9th logit gap is
**1.6–2.2** in bf16 logit units, which is enormous next to a 1-ULP input perturbation. The mechanism is real but
it is far too rare to produce the *same* 0.99457 on every rung of the ladder, which is what the artefacts record.

⚠ **What this test does not establish: the cost of a flip.** The model folds *two* immutable scale factors into
`router_combined_scale` and only one (`router.scale`) is recoverable from the checkpoint, so the softmax
temperature is wrong — the top-8 weights came out one-hot (1.0000 / 0.0000 ×7), which is an artefact of the
missing factor, not a finding. **The flip *rate* is valid** (a global scalar cannot change a `topk` ranking); the
flip *price* is not. For reference, a 5.055 × 10⁻³ drop implies a displaced routing weight of ≈ **0.071**.

### ⚠ 4. Sharded-at-all. Real, identical for every grid — and therefore inconsistent with v2.

The one thing that *does* change the norm output is going sharded at all: **1 bf16 ULP in 8–19 % of channels at
realistic activation magnitudes, zero at unit scale.** But it is identical for 2, 4, 8, 11, 22, 44 and 88 cores.
So if the whole-layer drop is caused by the norm change, it is caused by sharding at all — and then **v2's
88-core `decode_pcc = 0.99962934`, which shows no drop against v3's untouched incumbent 0.99962801, contradicts
it.** Both cannot be right.

---

# What the artefacts say, and this is the finding

`doc/advisor_challenger/build_evidence.py`, committed in the cell's own output:

```python
best = load(f"ladder_{11 if short == 'sliding' else 8}_{short}.json")
repeats = best["repeats_ms"]
passed = kind == "full_attention"
pcc = 0.9945729603715616 if not passed else 0.9997999978731844
incumbent_pcc = 0.9996280142258483 if not passed else 0.9997835173813693
```

Three things follow, in increasing order of seriousness:

1. **The timings are loaded from measurement files; the PCCs are literals.** `median_ms` and `repeats_ms` come
   from `ladder_11_sliding.json`. `oracle_pcc` and `incumbent_pcc_vs_reference` come from the source code.
2. **`oracle_passed` is the expression `kind == "full_attention"`.** The correctness verdict for the sliding kind
   is not derived from its PCC, or from anything else — it is a constant keyed on the layer kind, written next to
   the constant it is supposed to have been derived from. **The veto is hardcoded.**
3. **There is no oracle log for this cell.** No `oracle/` directory, no `pcc_*.json`, nothing to trace
   0.9945729603715616 to. v2's equivalent cell committed
   `oracle/norm88/pcc_layer0_sliding_attention_shared1.json` with its scope
   (`decode_current_pos=32, sequence_length=32, shared_physical_cache=true`) and a separate `prefill_pcc`.

**The fair reading of intent:** 0.9945729603715616 has sixteen significant digits and is not the sort of number
anyone invents. Almost certainly a real oracle run produced it and the agent hardcoded the observation into a
recorder script because there was no machine-readable oracle output to parse. **That does not change the
consequence:** nothing in the committed artefacts establishes that the oracle was ever run on the 11-core sliding
candidate, and the four experiments above say the candidate cannot have produced that number by the mechanism
claimed. **The gate passed the cell anyway** — because `oracle_pcc` and `op_under_test` were advisory, which is
[`PCC-BY-GRID`](ADVCHAL-V3-PCC-BY-GRID.md) §3 arriving at full price.

## And a second provenance defect found in the same read

The cell's `README.md` states:

> *"The full legal ladder was covered: 1 is the frozen incumbent, **2/4/8/11/22/44/88** were fresh processes.
> Because the advised value 88 is the legal maximum, there is no legal upper-side rung."*

and `build_evidence.py` writes `legal_ladder: [1, 2, 4, 8, 11, 22, 44, 88]`. **There is no `ladder_88_sliding.json`
or `ladder_88_full.json` in `measurements/`** — the directory holds 2, 4, 8, 11, 22 and 44 for each kind and
nothing else.

Either the 88 rung ran and its measurement file was never written, or the README overstates the sweep. **The
artefacts cannot distinguish, and no gate check compares a README's claimed measurements against the files.**
This also means my own earlier statement — *"88 was never tried"* — rests on absent files while the cell's prose
asserts the opposite; the honest position is that **the run's own two records disagree.**

The README does corroborate §3.2a in the cell's own words, which is worth having:

> *"All sliding ablations and ladder points inherit that kind-level veto; **only the fastest rung was spent on the
> absolute oracle**."*

---

# Conclusion

| candidate | verdict | evidence |
|---|---|---|
| the core count (11 vs 88) | **ruled out** | bit-identical output, every scale |
| v2-vs-v3 weight placement | **ruled out** | bit-identical output, real weight |
| expert-routing flips | **too rare** | 2 in 168 trials, 1 slot of 8, logit gap 1.6–2.2 |
| sharding at all | **real but inconsistent** | 1 ULP in 8–19 % of channels, same for all grids — contradicts v2's 0.99963 at 88 cores |
| **the number's provenance** | **this is the answer** | `oracle_pcc` hardcoded; `oracle_passed = kind == "full_attention"`; no oracle log committed |

**The −6,055 µs/model was not lost to a correctness problem. It was lost to a verdict with no traceable
measurement behind it, on an op that provably cannot produce the deviation it was blamed for.**

# Actions

1. **`oracle_passed` must be computed from `oracle_pcc`, and `oracle_pcc` must be parsed from a committed oracle
   artefact.** A literal in a build script cannot satisfy either. Gate: CRITICAL, per screened candidate.
2. **Commit the oracle log with its scope**, as v2 did — position, sequence length, cache mode, prefill and
   decode separately. Without the scope, two runs' PCCs are not comparable, which is RUN-LOG P4.
3. **Gate check: every measurement a README or evidence file claims must have a file in `measurements/`.** Catches
   the missing 88 rung mechanically.
4. **Re-run gemma-4-26B `-onA`'s sliding kind** with 1–3 in place. The candidate is 1.581980 ms against a
   1.824205 ms incumbent, **−242.2 µs/layer × 25 layers**, and it is currently default-off on the strength of a
   hardcoded constant.
5. **Assert `oracle_weights: "real"` against the checkpoint on disk** — see
   [`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md) §5. Layer-0 weights are fetchable by byte range in seconds,
   so "the weights are absent" is not a reason to weaken an oracle.
6. **Retire the routing hypothesis as the explanation**, but keep the finding: a `topk` over 128 experts flips on
   ~1 % of tokens under a 1-ULP perturbation, so a whole-layer PCC bar on this model has a ~1 % discontinuous
   floor regardless of the change under test. That is still a reason to gate placement on the op's own output.
