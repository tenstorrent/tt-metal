# Candidates measured 2026-08-20

Six experiments against a 9.311 ms baseline. Three pay, three do not.

**Outcome, decided 2026-08-21.** Two landed and have their own stage reports:
**B2 → change 30** ([30-aligner-bfp8-intermediate.md](30-aligner-bfp8-intermediate.md)) and
**A → change 31** ([31-bfp8-residual-last12.md](31-bfp8-residual-last12.md)), the latter at a
**12**-block suffix rather than the 18 measured here — 18 leaves only 1.5e-3 of the
`test_vision_transformer` gate, which prices out every later encoder change. **B1 was measured,
pays, and was deliberately not taken** on the same reasoning applied to a 0.9999 gate; it is now
a [DEAD_ENDS row](DEAD_ENDS.md#measured-and-rejected). Interleaving the residual instead of
suffixing it was measured on 2026-08-21 and **fails the gate** —
[DEAD_ENDS § Why it has to be a suffix](DEAD_ENDS.md#why-it-has-to-be-a-suffix).

Measured on a Wormhole N150, Release build, the harness in
[PERF.md's How to reproduce](../PERF.md#how-to-reproduce). Every delta below is against the
Step 0 baseline, not against the previous experiment — `perf_stage_report`'s "change from the
previous stage" column chains by stage number, which is wrong for a set of independent
one-at-a-time trials.

## Baseline, and a harness regression that had to be fixed first

`62c99c24f25` — the current HEAD, subject *"Add vision model profiling test and adjust device
parameters"* — sets

```
PERF_WARMUP_ITERS = 1
PERF_MEASURE_ITERS = 1
DEVICE_PERF_ITERS = 1
```

against `10 / 100 / 10` in every commit before it. Kernel time in this document and in PERF.md
is **the mean of replays 2-10**; with one replay `perf_stage_report` exits with
`expected several replays, found 1 occurrences of '768 x 1024'` and there is no comparable
figure at all. The counts were restored to `10 / 100 / 10` for every measurement here.

With them restored the baseline reproduces exactly: **9.311 ms, 293 ops** against the 9.316 ms
in PERF.md's header and the 9311.3 us of the reference profile.

Run costs, both wall-clock including post-processing: **one perf run 84-113 s**, **one accuracy
run 33-42 s**, all four accuracy gates in sequence **160-185 s**.

## Results

| step | change | kernel | Δ ms | Δ % | aligner (0.9999) | layernorm (0.99) | transformer (0.99) | tower (0.95) | verdict |
|---|---|---:|---:|---:|---|---|---|---|---|
| 0 | baseline | 9.311 | — | — | 0.999955 | 0.999972 / 0.999970 | 0.998760 | 0.970880 † | trusted |
| **A** | bfloat8_b residual, layers 6-23 | **9.222** | **−0.089** | **−0.96%** | 0.999955 | 0.999972 / 0.999970 | 0.991522 | 0.964073 | **best measured; see caveat** |
| **B1** | aligner `hifi2` → `hifi2_fp16` | **9.275** | **−0.036** | **−0.39%** | **0.999910** | 0.999972 / 0.999970 | 0.998808 | 0.970859 | **pays; spends 82% of the aligner margin** |
| **B2** | aligner intermediate → bfloat8_b | **9.294** | **−0.017** | **−0.18%** | 0.999940 | 0.999972 / 0.999970 | 0.998798 | 0.970875 | **pays, cheapest in accuracy** |
| C | layer norm HiFi4 → HiFi2 | 9.298 | −0.013 | −0.14% | 0.999955 | 0.999967 / 0.999965 | 0.998741 | **0.962398** | reject — worst PCC-per-us here |
| C′ | layer norm HiFi4 → LoFi | — | — | — | — | — | — | **0.937403 FAIL** | reject |
| B3 | B1 + aligner `out_subblock_w` 4 → 8 | 9.277 | +0.002 vs B1 | +0.02% | not run | not run | not run | not run | reject — flat |

† the baseline tower figure is PERF.md's; it was not re-measured notrace this session. B1 and B2
both read 0.97086-0.97088 with the tower untouched, which brackets it.

## A — bfloat8_b residual, partial by layer

The full-24 version is [DEAD_ENDS' largest deliberately-absent win](DEAD_ENDS.md#bfloat8_b-residual-stream).
What had never been tried is anything between all and none. It is not a binary: **PCC is
monotonic in the number of bfp8 layers**, so there is a boundary, and it is at 18.

| last N layers bfp8 | `JANUS_BFP8_RESIDUAL_FROM` | transformer PCC | vs 0.99 |
|---:|---:|---|---|
| 0 | unset | 0.998760 | pass |
| 6 | 18 | 0.997891 | pass |
| 12 | 12 | 0.996626 | pass |
| **18** | **6** | **0.991522** | **pass, margin 1.5e-3** |
| 19 | 5 | 0.989845 | **fail by 1.5e-4** |
| 20 | 4 | fail | fail |
| 21 | 3 | fail | fail |
| 22 | 2 | fail | fail |
| 24 | 0 | fail | fail (0.9765 per DEAD_ENDS) |

Strictly decreasing at every point measured, so bisection was sound. Nine accuracy runs, ~6
device-minutes.

The gain arrives through the documented mechanism and **op count does not move** (293 either
way) — the norms inherit the format, so there are no typecasts, which is what separates this
from [stage 26's standalone narrowing](DEAD_ENDS.md) at +0.396 ms.

| op | baseline us | A us | Δ each | Δ total ms |
|---|---:|---:|---:|---:|
| `c_fc` 576x1024x4096 | 72.30 | 70.80 | −1.50 | −0.036 |
| `qkv` 576x1024x3072 | 48.80 | 47.70 | −1.10 | −0.026 |
| LayerNorm | 19.23 | 18.82 | −0.41 | −0.020 |
| BinaryNg (the adds) | 3.06 | 2.83 | −0.23 | −0.011 |

Two thirds of it is the `qkv` and `c_fc` in0 multicast halving, exactly as predicted; the adds
themselves are a ninth of it.

**Recalibration worth keeping.** Stage 11 measured the all-24 form at **−8.70%**. Scaling this
result to 24 layers gives roughly −1.3%. The lever did not shrink — the tower did: stage 11 ran
at 15.88 ms on span, before bfp8 weights, LoFi and the output sharding. **A −8.70% result at
15.88 ms is not a −8.70% result at 9.31 ms**, and this is the second entry in the campaign where
a percentage failed to survive its denominator.

**Why it is a caveat and not a recommendation.** "Layers 6-23" is a magic constant with one PCC
measurement behind it. Per-layer dtype variation is first-class infrastructure in this repo
(`get_tensor_dtype(decoder_id=...)`, `models/tt_transformers/tt/model_config.py:4461`) but every
shipped config — accuracy, performance, every per-model exception — applies one dtype to all
layers; variation in the repo is per *tensor*, never per *layer index*. 18 is not expressible as
a one-sentence rule, it is where this gate happened to break on this image. It also spends most
of the transformer gate's headroom (1.5e-3 left) to buy 0.96%.

## B1 — the aligner's fp32 dest accumulation

`compute_kernel_config_hifi2` → `compute_kernel_config_hifi2_fp16`
(`models/tt_transformers/tt/model_config.py:790-801`). Same fidelity; `fp32_dest_acc_en`
True → False and, coupled in the same constant, `math_approx_mode` True → False.

| matmul | baseline us | B1 us | Δ |
|---|---:|---:|---:|
| aligner fc1 576x1024x4096 | 235.8 | **213.4** | −22.4 (−9.5%) |
| aligner hidden 576x4096x4096 | 328.3 | **310.4** | −17.9 (−5.5%) |

−40.3 us of aligner against a −36 us tower delta; every body matmul inside ±0.3 us. The two
coupled flags did **not** need isolating: the aligner's fused GELU carries its own
`APPROXIMATION_MODE` explicitly at False in `_FUSED_ACT`
(`janus_pro_vision_aligner.py:14-22`), so `math_approx_mode` has no gelu to act on here, and
both time and PCC moved in one direction.

**The cost is the aligner gate.** 0.999955 → 0.999910 against 0.9999 — margin 5.5e-5 → 1.0e-5,
i.e. **82% of the remaining headroom for 0.39% of tower time**. It passes. It also means the
next aligner change measures against almost no slack, and B1+B2 together were deliberately not
measured for that reason.

## B2 — the aligner's intermediate in bfloat8_b

fc1's output has exactly one consumer, so it narrows under the campaign's read-once rule.
Placement stays DRAM ([L1 was +24 us at stage 25](DEAD_ENDS.md)); only the dtype changes.

| matmul | baseline us | B2 us | Δ |
|---|---:|---:|---:|
| aligner fc1 | 235.8 | 230.9 | −4.9 |
| aligner hidden | 328.3 | 314.9 | −13.4 |

**Cheaper in accuracy than B1 and it was expected to be dearer** — aligner 0.999940 against
B1's 0.999910, so narrowing a whole 4.72 MB tensor to 8 bits costs a third of what turning off
fp32 accumulation costs. Halving the bytes was the smaller numerical event.

**And it barely pays, which is the more useful half of the result.** fc1's reported DRAM
utilisation falls 13.1% → 9.9% — the write did halve — and its time moves 2%. The two ops are
classified writer-bound, the write is now half the bytes, and nothing much happened. That is the
transaction-and-issue-overhead thesis of
[PROFILER_NOTES](PROFILER_NOTES.md#dram-bandwidth-is-not-the-limit) holding on the one module
that had never been tested against it.

## Closing PROFILER_NOTES' open item

[PROFILER_NOTES.md](PROFILER_NOTES.md#what-the-profiling-did-not-establish) listed the aligner as
uncharacterised. B1 and B2 characterise it, and the 3.27x gap against `c_fc` at the identical
shape and program config resolves into three named parts:

| difference | worth |
|---|---:|
| `fp32_dest_acc_en` True → False (halves the DST budget) | **−22.4 us** |
| output bfloat16 → bfloat8_b | −4.9 us |
| fidelity HiFi2 vs LoFi | not measured — the aligner feeds the language model |

fc1 at 235.8 us against `c_fc` at 72.3 us is therefore mostly **fp32 dest accumulation**, not
fidelity and not the output format. `out_subblock_w` is not part of it (B3, below).
