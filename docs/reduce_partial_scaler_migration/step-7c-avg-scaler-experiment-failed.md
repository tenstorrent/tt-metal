# Step 7c — using the AVG scaler for the layernorm mean: **failed**

Experiment on top of `malimpic/experimental-reduce-helpers-migration-layernorm-2`.

**Outcome: rejected. 3 of 265 layernorm tests fail, no performance benefit.** The code change is not
committed; this document is.

## What was tried

layernorm computes its mean as `reduce<PoolType::SUM>` with a scaler tile of exactly `1.0`, then
divides by `N` once at the end via an fp32 `mul_unary_tile` in the post-reduce op. The obvious
simplification is to let the reduce do the division: use `PoolType::AVG`, whose scaler tile carries
`1/N`, and drop the epilogue entirely.

Changes made (all reverted):

| File | Change |
|---|---|
| 3 × `reader_unary_interleaved_ln*.cpp` | scaler `SUM`/`1.0` → `AVG`/`1/W` |
| `numeric.h` | `row_wise_mean` and `row_wise_mean_with_pre_add` skip the `1/N` epilogue when `reduce_type == AVG` |
| `layernorm.cpp`, `layernorm_large_tensor.cpp` | 4 call sites `PoolType::SUM` → `PoolType::AVG` |
| `layernorm_large_tensor.cpp` | removed the `mul_unary_tile(1/W)` from the open-coded variance loop |

That last one is worth noting as a trap: the open-coded variance reduce at `large_tensor.cpp:202`
shares the same scaler CB and applied its **own** `1/W` afterwards. Switching the shared scaler to
AVG without touching it would have divided by `W` twice — silently, since no static check covers it.

## Why it fails

The scaler CB is `Float16_b` (`layernorm_op_multi_core.cpp:164`). `1.0` is exactly representable
there; `1/W` generally is not. With SUM the reduce contributes zero quantization error and the
divisor is fp32 (24-bit mantissa). With AVG the divisor is quantized to bfloat16's 8 mantissa bits,
and — critically — the error is **systematic**: every element is scaled by the same wrong constant,
so it biases `E[x]` and `Var[x]` rather than averaging out.

Of the 19 widths the layernorm suite parametrises, only the 5 powers of two survive exactly.

## Test failures

`3 failed, 262 passed`. All three are the same width:

```
FAILED test_layer_norm_with_padding[dtype=torch.float32-use_welford=False-w=487-h=32]
FAILED test_layer_norm_with_padding[dtype=torch.float32-use_welford=False-w=487-h=2999]
FAILED test_layer_norm_with_padding[dtype=torch.float32-use_welford=False-w=487-h=2066]
```

| case | metric | measured | threshold | over by |
|---|---|---:|---:|---:|
| w=487, h=32 | relative Frobenius norm | 1.350368e-02 | 1.05e-02 | **+28.6%** |
| w=487, h=2999 | relative Frobenius norm | 1.351165e-02 | 1.05e-02 | **+28.7%** |
| w=487, h=2066 | relative Frobenius norm | 1.351130e-02 | 1.05e-02 | **+28.7%** |

Each failed 1 of 3 numeric checks (Frobenius); PCC and allclose still passed.

**W=487 is exactly the worst-case width in the table** — `1/487` has a 0.3189% bf16 representation
error, the largest of the 19 tested. The failure is width-dependent and H-independent (the same
~1.351e-02 at h=32, 2066 and 2999), which is the signature of a constant-scaling error rather than an
accumulation error.

Note also that all three failures are `dtype=torch.float32`. With fp32 input the rest of the pipeline
is more accurate, so the bf16 scaler is a larger share of the total error and there is less headroom
under the threshold.

## Precision measurement

End-to-end mean absolute error against a float64 reference, same seed and shapes:

| W | 1/W in bf16 | SUM + fp32 divide | AVG scaler | change |
|---:|:---:|---:|---:|---:|
| 512 | exact | 0.002102 | 0.002102 | bit-identical |
| 1024 | exact | 0.002065 | 0.002065 | bit-identical |
| 2048 | exact | 0.002006 | 0.002006 | bit-identical |
| 3072 | lossy | 0.002842 | 0.006387 | **+125%** |
| 4096 | exact | 0.002851 | 0.002851 | bit-identical |
| 6144 | lossy | 0.003006 | 0.006762 | **+125%** |
| 8192 | exact | 0.002797 | 0.002797 | bit-identical |

Bit-identical on every power of two, error more than doubling on every non-power-of-two. This is as
clean a confirmation of the mechanism as the data could give: the only thing that changed is where
`1/N` lives, and the effect appears exactly when `1/N` stops being representable.

## Performance

No benefit. Removing one `mul_unary_tile` per output row is lost in the noise:

| case | SUM + fp32 | AVG scaler | delta |
|---|---:|---:|---:|
| `layernorm.aligned_4096` | 648.9 | 649.8 | +0.14% |
| `layernorm.ragged_4095` | 649.6 | 650.1 | +0.08% |
| `layernorm_wb.aligned_4096` | 933.6 | 933.9 | +0.03% |
| `layernorm_wb.ragged_4095` | 935.4 | 934.4 | −0.11% |
| `layernorm_residual.aligned_4096` | 1094.8 | 1093.4 | −0.13% |
| `layernorm_residual.ragged_4095` | 1096.5 | 1098.0 | +0.14% |

All within ±0.15%, i.e. inside noise. The epilogue is one SFPU op per output tile, against a whole
row of `reduce_tile` work — there was never much to win.

## Conclusion

The existing SUM-plus-fp32-divide design is correct and should stay. It costs one SFPU op per row
(unmeasurable) and buys an exactly-representable scaler plus a 24-bit divisor.

This also calibrates the earlier speculation in step 7b. The measured penalty for the AVG scaler is
~2.2× the error on non-power-of-two widths, enough to break a real test by 29%. For comparison, the
`Accumulate`-chain penalty measured in step 7b was 3–10% of the error and broke nothing. So of the two
precision compromises considered in this migration, this is the one that actually matters — and it is
the one that was already avoided.

### Related: `moreh_mean_h` does make this trade

`moreh_mean_h` uses `PoolType::AVG` with `reduce_factor = origin_H` and a bf16 scaler, so it takes the
quantization this experiment rejects. Measured there (mean over H, bf16 input, error vs the float64
mean of the same bf16 values):

| H | 1/H bf16 err | measured mean err | signed bias |
|---:|---:|---:|---:|
| 512 (pow2) | 0.0000% | 0.6423% | −0.5924% |
| 487 | 0.3189% | 1.1058% | −1.0888% |

Its accumulation error (0.26–0.83%, present even at exact powers of two) is the dominant term, so the
scaler quantization is not the whole story there — but it is additive and systematic. `moreh_mean_h`
passes its tests because they use `rtol = atol = 0.05–0.12`, tolerances two orders of magnitude looser
than layernorm's Frobenius threshold. Whether it should be switched to the layernorm pattern is a
separate question this experiment does not answer.
