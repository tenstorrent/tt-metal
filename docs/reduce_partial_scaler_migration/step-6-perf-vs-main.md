# Step 6 — performance vs `main`

Device-time comparison of every op touched by Steps 1–3, measured on the same machine, same
submodule pins, alternating branch builds.

## Method

`tests/ttnn/unit_tests/operations/moreh/bench_reduce_partial_scaler.py`. Enqueues a batch of `ITERS`
ops and synchronises once at the end, so per-call host overhead is amortised; reports the best of
`REPEATS` batches after warmup.

Shapes were deliberately scaled up until device work dominates. An earlier pass with small shapes
(`[3, 2, 320, 320]`, `[1, 1, 64, 32]`) put **every** case in a 33–39 µs band — that was dispatch
overhead, and it would have hidden any real kernel difference. The reported shapes put the ops at
129–652 µs.

Every case is run in a tile-aligned **and** a ragged variant. The migration only restructures the
ragged path, so the aligned rows act as a control.

`main` was measured twice to establish the noise floor before any delta was believed.

## Results

Microseconds per op, mean of **two runs per branch**, with each branch's own run-to-run spread shown.

| case | branch | main | delta |
|---|---:|---:|---:|
| `moreh_sum_h.aligned_1024` | 348.7 ±0.89% | 348.7 ±0.27% | −0.01% |
| `moreh_sum_h.ragged_1023` | 347.0 ±1.59% | 364.2 ±0.23% | **−4.71%** |
| `moreh_mean_h.aligned_1024` | 348.8 ±0.05% | 351.5 ±0.45% | −0.76% |
| `moreh_mean_h.ragged_1023` | 351.3 ±1.34% | 365.7 ±1.12% | **−3.93%** |
| `softmax_small_h.aligned_512` | 178.7 ±0.32% | 179.8 ±0.14% | −0.63% |
| `softmax_small_h.ragged_511` | 178.5 ±0.14% | 178.9 ±1.58% | −0.26% |
| `softmax_small_w.aligned_512` | 129.9 ±0.50% | 127.2 ±0.07% | **+2.18%** |
| `softmax_small_w.ragged_511` | 128.6 ±0.43% | 125.3 ±0.27% | **+2.61%** |
| `layernorm.aligned_4096` | 652.9 ±0.22% | 652.4 ±0.11% | +0.08% |
| `layernorm.ragged_4095` | 651.5 ±0.14% | 653.4 ±0.09% | −0.29% |

Run-to-run noise is 0.1–0.5% for most cases (worst 1.6%), so deltas above ~2% are real.

## Reading the results

**moreh sum/mean: ~4% faster on ragged shapes, and the ragged penalty disappears.** This is the
headline result and it is exactly the shape of win the migration predicted:

| | aligned | ragged | ragged penalty |
|---|---:|---:|---:|
| `main` | 348.7 | 364.2 | **+4.4%** |
| branch | 348.7 | 347.0 | **−0.5%** |

On `main`, a ragged H cost 4.4% more than an aligned one — that was the price of the workaround
(masking the last tile through DST, packing it to a scratch CB, and a second accumulating reduce).
On the branch a ragged reduce costs the same as an aligned one. The partial scaler is free; the
workaround was not.

**layernorm: unchanged**, within 0.4% on both variants. Correct — Step 3 was a pure reader refactor
emitting byte-identical CB contents, and the measurement confirms it.

**softmax_small_w: a real ~2–3% regression.** It is outside the 0.1–0.3% noise for those rows and it
appears on **both** the aligned and ragged variants, so it is not a ragged-path effect.

## The softmax_w regression

Cause not confirmed by profiling; the leading candidate is the price of shape-genericity. Because
moreh softmax gets `mask_w` as a *runtime* value, the migrated kernel does the same thing for every
shape: the reader always emits **two** scaler tiles and the compute always selects tile 1 for the
last tile. On `main`, an aligned shape emitted one tile and always read index 0. So the branch pays
one extra scaler-tile fill per op regardless of whether the shape is ragged.

That is consistent with the regression showing up on the aligned variant too, and with it being worse
in the `w` case (129 µs of total work) than the `h` case (178 µs), where the same fixed cost is a
smaller fraction. It does **not** fully explain why `softmax_small_h` is marginally *faster*, so this
should be treated as a hypothesis until someone profiles it.

### How to remove it

`ReducePartialScaler` is a runtime struct, so the kernel can choose per launch instead of paying
unconditionally:

```cpp
const auto partial = (mask_w < TILE_WIDTH) ? ReducePartialScaler::last_tile_at(1)
                                           : ReducePartialScaler::none();
```

This needs `mask_w` plumbed to the compute kernel (it is currently only a reader runtime arg), and
the reader would emit one tile in the aligned case. Left undone rather than bolted on at the end of
the migration: it is a small, self-contained follow-up that deserves its own measurement.

## Verdict

No regression in the two places the migration was aimed at. A clear ~4% win on ragged moreh
reductions, and the ragged-vs-aligned penalty eliminated entirely. One real but small (~2–3%)
regression on `softmax_small_w`, with a known likely cause and a concrete fix.

Reproduce with:

```
python tests/ttnn/unit_tests/operations/moreh/bench_reduce_partial_scaler.py out.json
```
