# Device Gumbel restored: what was fixed, what was not, and how to tell them apart (#48291)

**Status 2026-07-26: the RNG-driven canvas collapse is fixed and `DG_VLLM_GUMBEL_MODE=device` is
the default again.** The residual collapses are a different mechanism with a different owner —
they are the #48291 fidelity gap, not the sampler. This document exists mostly so that the second
mechanism is not mistaken for the first a third time.

## 0. Why device at all

`host` Gumbel is correct but does not meet the throughput bar: **~53.6 vs ~36.3 tokens/block/s**
measured at 4096 on a matched 4-seed A/B (~1.48x). Fixing the device path was therefore the task,
rather than routing around it.

## 1. Two defects, both measured, both fixed

### 1a. `ttnn.rand` is a sliding window over one stream

Mapping one 32x32 rand tile element by element: **94 distinct values in 1024 slots**. The PRNG is
per-lane (32/32 distinct inside one 32-lane SFPU vector), but only 20 of a tile's 32 vector draws
were distinct, with the exact relation

    (face f, vector 2k) == (face f+1, vector 2k+1)

which in tile coordinates is "column c is byte-identical to column c-24 for c % 32 >= 24". So
element `(read t, lane i)` carries `stream[t + i]`: one window advancing about one element per read
while all 32 lanes read overlapping positions.

Fix: consume several PRNG values per stored element
(`tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_rand.h`).

| | tile distinct | distinct vector draws | byte-identical rows | max abs r | max argmax mult |
| --- | --- | --- | --- | --- | --- |
| before | 94/1024 | 20/32 | 64/256 | 1.00000 | 11 |
| after | 214/1024 | 32/32 | **0/256** | 0.618 | 5 |
| host IID | — | — | 0/256 | 0.035 | 2 |

Distribution untouched: mean 0.4994, std 0.2887, top decile 0.0996 over 524288 samples.

**Measured dead ends — do not repeat.** NOP spacing after the PRNG read (0 through 32: byte-identical
output, so not a pipeline hazard, and the Wormhole kernel's extra NOPs are irrelevant here);
xorshift32 over two draws (no gain — any combination of reads is still a function of `t + i`);
`SFPTRANSP` across four draws with an XOR fold (modest, and it re-introduced duplicate rows). One
trap: holding `scale`/`from` in lreg4/lreg5 across a transpose silently breaks the output range to
mean 0.35 / std 0.64, which passes the correlation metrics by destroying the distribution instead.

### 1b. DG was drawing the noise on the wrong axis

`sample_gumbel_noise_with_permuted_vocab` keeps vocab off the rand innermost axis by collapsing
every OTHER axis into it — and for the production shape `(1, 1, 256, vocab)` that is the 256 canvas
positions. It moved the degeneracy from an axis where it biases WHICH token a position picks onto
the axis where it makes DIFFERENT POSITIONS pick the same token, which is what collapses a canvas.

Measured at the production geometry (canvas 256, vocab 262144), with the kernel already fixed:

| layout | max abs r | distinct flat-logit winners | max multiplicity |
| --- | --- | --- | --- |
| permuted (was production) | 0.598 | 154/256 | 8 |
| **vocab innermost (now)** | 0.350 | **253/256** | **2** |
| host IID control | 0.009 | 256/256 | 1 |

## 2. The two collapse mechanisms, and the telemetry that separates them

Both end with a committed canvas dominated by one token, so the token counts alone cannot tell
them apart. The halt telemetry can, and they have different owners:

| | CONVERGED | NON-CONVERGED |
| --- | --- | --- |
| entropy trajectory | 4.3–5.1 → **~0.01** | 3.6 → **2.4**, stays high |
| `halt_mismatch_final` | **0** (argmax settled) | **109** of 256 still flipping |
| `halt_blocking_gate` | `entropy` | `both` |
| meaning | converged onto a confident degenerate fixed point | step cap hit mid-churn |
| cause | correlated noise across positions | fidelity gap vs the reference |
| owner | **fixed here** | **#48291** |

## 3. The configuration is faithful, so NON-CONVERGED is not a settings bug

Every knob matches the released `generation_config.json` exactly, checked item by item:

| reference | DG |
| --- | --- |
| `max_denoising_steps: 48` | `max_denoise_steps = 48` |
| `confidence_threshold: 0.005` | `entropy_stop_threshold = 0.005` |
| `stability_threshold: 1` | `stable_steps_to_halt = 1` |
| `t_max: 0.8` / `t_min: 0.4` | `temperature_start = 0.8` / `temperature_end = 0.4` |
| `sampler_config.entropy_bound: 0.1` | `entropy_budget = 0.1` |

The CUDA reference converges within the same 48 steps on the same prompts and never collapses
(0/198). Same budget, same thresholds, same schedule, different outcome — that is a numerical
fidelity difference, i.e. #48291.

## 4. The CUDA reference arm (what is inherent, what is ours)

Full GPQA-Diamond, thinking, through the CUDA vLLM server — same model, no TT sampler anywhere:

| | @8192 | the 45 unanswered, retried @16384 |
| --- | --- | --- |
| accuracy | 124/198 (62.6%) | 24/45 rescued → **148/198 (74.7%)** |
| answered | 153/198 (77.3%) | 40/45 (88.9%) |
| COLLAPSE (>=20 identical consecutive tokens) | **0/198 (0.0%)** | — |
| tail repetition | 44/198 (22.2%) | 22/45 (48.9%) |
| no answer at the cap | 45/198 (22.7%) | 5/45 (11.1%) |

Three things follow:

* **Truncation and tail repetition are inherent.** The GPU truncates 22.7% at 8192 and repeats in
  22.2% of answers. Calling those "TT degeneration" was a category error.
* **Canvas collapse is NOT inherent** — the GPU never does it. That mode was ours, and it is the
  one fixed above.
* **Roughly a third of the "unanswered" set is noise, not budget.** Of the 40 rescued at 16384,
  **16 answered in under 8192 tokens** — the first run simply took a looping trajectory. vLLM's
  continuous batching is numerically batch-dependent, so even at temperature 0 the same question
  can go either way. Any truncation rate measured with batching carries that variance.

## 5. Device gate

Matched 4-seed A/B, shipped serving configuration, GPQA doc0 in thinking mode: 8/8 correct on both
arms, guard never fired, device ~1.48x faster. The two documents that previously failed hardest —
doc6 and doc9, both of which emitted ZERO characters because the guard fired on block 0 — now
generate normally, and doc9 answers `B`, matching the CUDA reference.

A full 198-question run at 16K is in progress (`tmux gpqa198` on the QB2 box; resumable, skips
questions whose metrics JSON already exists). Interim at 42/198: the CONVERGED mechanism appears
**once**, stable across every checkpoint (n=11, 22, 30, 36, 42), while NON-CONVERGED accounts for
the other 17. That split is the result this work is accountable for; the final aggregate numbers go
here when the run lands.

Note a real trade-off visible at 16K: truncation goes to 0%, but the collapse rate rises (30% at
8192/12 blocks vs ~43% at 16K/60 blocks) because a larger budget gives the NON-CONVERGED mode far
more blocks in which to churn. Raising the context is not free until #48291 is addressed.

## 6. Reproduce

```bash
# RNG health, kernel level (fails on the pre-fix kernel)
DG_RUN_DEVICE=1 pytest tests/ttnn/nightly/unit_tests/operations/rand/test_rand_independence.py -s

# canvas-position independence of the DG draw
DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_gumbel_position_correlation.py -s

# which halt gate blocked, per block, on any traced run
grep DG_TRACE_METRIC <log> | grep upfront_replay   # halt_blocking_gate, halt_entropy_*, halt_mismatch_*

# degeneracy statistics per committed canvas
grep DG_DEGENERACY <log>
```
