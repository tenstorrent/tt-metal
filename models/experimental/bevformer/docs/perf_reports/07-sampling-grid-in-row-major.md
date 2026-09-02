# Stage: 07-sampling-grid-in-row-major

| | |
|---|---|
| commit | [`a32ddae6c62`](https://github.com/tenstorrent/tt-metal/commit/a32ddae6c62363ba9ad45844a8d08e8655d564b4) |
| candidate | [5b](../perf_optimization_candidates.md#5b-sampling-location-math-in-row_major) |
| config | `nuscenes_base`, 100×100, N150 |
| profile | **356.2 ms kernel**, 131 ops (**+5**), CSV `generated/profiler/reports/2026_09_01_23_33_36/` |
| delta | **−82.2 ms kernel (−18.8%)** vs [stage 06](06-sca-key-permute-deleted.md)'s 438.4 ms; **−324.6 ms (−47.7%)** cumulative from stage 03 |
| PCC | **0.999651**, up from 0.999611 — [expected](#pcc-improved-and-that-is-expected) |
| suite | `tests/pcc/` **33 passed, 0 failed** — the 200×200 case that had failed since before stage 04 now passes |

## What changed

Every step between the `sampling_offsets` Linear and the fused op ran in TILE on a tensor whose
trailing axes were `(num_points, 2)` or `(num_levels, num_points, 2)` — extent 4 and 2, padded to
`32 × 32`. Six ops, 104.4 ms across SCA and TSA, computing on 1/128 real data:

```
Linear → (bs, Q, heads*L*P*2)  TILE, 256 wide           0.25 ms   ← already a good shape
reshape → (bs*Q*heads, L, P, 2)                        16.88 ms
reshape → (bs, Q, heads, L, P//D, D, 2)                 4.44 ms
add reference_points                                   14.98 ms
mul 2.0 ; sub 1.0                                      19.09 ms
to_layout ROW_MAJOR                                    14.26 ms   ← reads 490 M padded elements
                                                                    to produce 3.8 M real ones
```
(SCA figures; TSA carried the same chain for another 34.8 ms.)

Every step is elementwise or a reshape, and elementwise ops do not care what the axes mean. So the
chain now runs on the `(bs, Q, 256)` shape the Linear already emits, in ROW_MAJOR:

```
Linear → (bs, Q, 256) TILE
to_layout ROW_MAJOR
add grid_bias                                           ← one add, on real data
reshape → (bs, Q, heads, L, P, 2)                       ← 7.09 ms, see the correction below
```

> **Corrected by [stage 09](09-head-major-sampling-grid.md).** That last reshape was booked as a free
> view. It is not: a ROW_MAJOR reshape is a view only when the **last dimension is unchanged**, and
> `256 → 2` changes it. It measured **7.09 ms**. The −82.2 ms here is real and unaffected — it was
> measured, not derived — but the reasoning was luckier than it deserved, and stage 09 collected the
> remaining 24.5 ms once the rule was right.

Two constant folds make the single add sufficient:

- **`2 / [W, H]` in the Linear weight.** `grid == 2·(ref + off) - 1 == (2·ref - 1) + 2·off`, so the
  `2·off` half multiplies into the scale [stage 05](05-offset-normalizer-folded.md) already put there.
  One constant, not two.
- **`2·ref - 1` as `grid_bias`**, laid out in the Linear's channel order. The Linear emits
  `(head, level, point, xy)` with points grouped `(P//D, D)`, and the reference point broadcasts over
  everything but the innermost `(D, 2)` block — so the bias is that block repeated once per
  (head, level, point-group), one ROW_MAJOR `repeat`.

The `mul`/`sub`/`to_layout` in `multi_scale_deformable_attn_ttnn` go with them; the function now takes
`sampling_grids` already normalized and ROW_MAJOR.

## Where the time went

| op | stage 06 | stage 07 | Δ |
|---|---:|---:|---:|
| BinaryNg | 49.3 ms | **2.3 ms** | **−47.0** |
| ReshapeView | 77.4 ms | 55.3 ms | **−22.1** |
| UntilizeWithUnpadding | 42.4 ms | 26.5 ms | **−15.9** |
| RepeatCodegen | 1.0 ms | 3.0 ms | +2.0 |
| everything else | 268.3 ms | 269.1 ms | +0.8 |

**`BinaryNg` went 49.3 → 2.3 ms** — candidate 5's opening claim confirmed: the arithmetic was never
expensive, the padding under it was. Seventeen instances still run, now at 0.14 ms each instead of up
to 15.

**Op count went up by 5 while kernel dropped 82 ms.** Candidate 5 was framed as "reduce the number of
ops"; what pays is reducing the padded bytes each op touches.

## PCC improved, and that is expected

0.999611 → **0.999651**. Not noise: the change removes two bfloat16 rounding steps from every sampling
coordinate (`mul` then `sub`, each rounding to bf16) and replaces them with constants folded at
float32 into the weight and the bias. Algebraically exact; only the rounding schedule differs.

## It also fixed the 200×200 OOM

`test_spatial_cross_attention.py::…[nuscenes_base-1-200-200-…]` had failed with a DRAM allocation
error since before stage 04, and both [1b](../perf_optimization_candidates.md#1b-bound-max_len-statically)
and its [stage-04 re-check](04-fused-msda.md) recorded it as the memory ceiling that made a static
`max_len` bound unaffordable — attributing it to "the sampling-location math upstream of the op".

That attribution was right, and this change is what touched it. **Verified by bisection:** with only
[5a](06-sca-key-permute-deleted.md) applied the case still fails; with 5b applied the suite reads
33 passed. The `(bs*Q*heads, L, P, 2)` intermediate padding to `32 × 32` was the allocation.

**This reopens 1b** on the memory argument. Its +129 ms cost argument stands on its own, but it can no
longer be dismissed on DRAM.

## Layer profile now

| Op | inst | ms | % | | Op | inst | ms | % |
|---|---:|---:|---:|---|---|---:|---:|---:|
| MSDAOperation | 5 | 167.9 | 47.1 | | Slice | 13 | 13.2 | 3.7 |
| ReshapeView | 21 | 55.3 | 15.5 | | Scatter | 1 | 10.5 | 2.9 |
| Permute | 10 | 43.7 | 12.3 | | Transpose | 12 | 7.9 | 2.2 |
| UntilizeWithUnpadding | 18 | 26.5 | 7.4 | | Matmul | 11 | 4.7 | 1.3 |
| TilizeWithValPadding | 6 | 17.2 | 4.8 | | RepeatCodegen + BinaryNg | 20 | 5.3 | 1.4 |

`MSDAOperation` is **47.1%** of the layer. The upstream questions
([10](../perf_optimization_candidates.md#candidate-10--msdaoperation-itself) /
[12](../perf_optimization_candidates.md#candidate-12--one-fused-call-for-all-levels)) are closing in on
being the only thing that matters.
