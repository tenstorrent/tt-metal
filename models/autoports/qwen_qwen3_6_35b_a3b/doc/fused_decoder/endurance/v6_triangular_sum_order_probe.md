# Exact CPU triangular row-update reduction order

**Resolved:** the independent reduction reproduces every FP32 bit on all63 actual layer34/seed154 triangular updates (32 heads × two chunks), plus189 randomized and signed-zero cases. The [result](v6_triangular_sum_order_probe.json) covers165,312 outputs with **zero differences**. The [CPU implementation](../../../tests/v6_triangular_sum_order_probe.py) exposes `canonical_outer_sum(products)` for contiguous `[..., i, i]` products, `1 <= i <= 63`.

The installed Torch git commit is `70d99e998b4955e0049d13a98d77ae1b14db1f45`. Its [primary SumKernel source](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cpu/SumKernel.cpp) supplies the missing ordering. The exact retrieved source is preserved in [the snapshot](v6_torch_SumKernel.cpp.snapshot), with URL and hash recorded in the result. Although the machine reports AVX512 capability, this sum kernel deliberately excludes AVX512 dispatch; its FP32 vector width is8.

For these shapes the output-column position determines the reduction:

| Logical row width i | Output columns | Required order |
| --- | --- | --- |
| 1–7 | Full groups of four columns | Sequential FP32 accumulation over the reduction axis |
| 1–7 | Remaining columns | Four-partial method below |
| 8–31 | All columns | Four-partial method |
| 32–63 | First32 columns | Sixteen-element cascade method below |
| 32–63 | Remaining columns | Four-partial method |

The **four-partial method** accumulates full groups of four reduction entries into four independent accumulators. Add every remaining entry to accumulator0, then add accumulator1, accumulator2 and accumulator3 to accumulator0 in that order. The earlier experiments distributed remainder entries among accumulators or combined partials before adding the remainder; both change rounding.

The **sixteen-element cascade method** sequentially sums each complete16-element block from zero, adds those block sums to a separate running total, and independently sums the remaining tail from zero. The final addition is **tail sum + completed-block total**. In these shapes the higher cascade levels remain zero. The implementation also reproduces accumulation into the initially zero output, including signed-zero behavior.

Both algorithms use separately rounded products already present in the triangular update. This reduction mapping does not resolve the distinct matrix-product FMA or cumulative-log-decay precision requirements documented in [the earlier ordering probe](v6_chunk_cpu_order_probe.md).

An initial attempt incorrectly inferred vector width16 from reported CPU capability. Its [failed result](v6_triangular_sum_order_avx512_assumption_rejected.json) and [source snapshot](v6_triangular_sum_order_avx512_assumption_rejected.py.snapshot) are retained. The corrected8-wide dispatch matches every tested actual and random output; no fixture-specific values or outcomes are hardcoded.

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_triangular_sum_order_probe
```

CPU only, four Torch threads. No hardware access, TTNN import, production edit, native/shared edit, or acceptance-gate change. This is an exact CPU reduction-order proof for the stated shapes and installed build, not evidence that a proposed device implementation is correct or fast.
