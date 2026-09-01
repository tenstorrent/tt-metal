# Stage: 07-sampling-grid-in-row-major

- candidate: [5b](../perf_optimization_candidates.md#5b-do-the-sampling-location-math-in-row_major)
- config: `nuscenes_base`, 100×100, N150
- layer profile: **356.2 ms kernel**, 131 device ops (+5), CSV
  `generated/profiler/reports/2026_09_01_23_33_36/`
- **−82.2 ms kernel (−18.8%)** against [stage 06](06-sca-key-permute-deleted.md)'s 438.4 ms;
  **−324.6 ms (−47.7%)** cumulative against the stage-03 layer
- PCC: **0.999651**, up from 0.999611 — see [below](#pcc-improved-and-that-is-expected)
- `tests/pcc/`: **33 passed, 0 failed** — the 200×200 case that had failed since before stage 04
  now passes, see [below](#it-also-fixed-the-200200-oom)

## What this change was

Every step between the `sampling_offsets` Linear and the fused op ran in TILE layout on a tensor
whose trailing two axes were `(num_points, 2)` or `(num_levels, num_points, 2)` — extent 4 and 2,
padded to `32 × 32`. Six ops, 104.4 ms across SCA and TSA, computing on 1/128 real data:

```
Linear → (bs, Q, heads*L*P*2)  TILE, 256 wide            0.25 ms   ← already a good shape
reshape → (bs*Q*heads, L, P, 2)                         16.88 ms
reshape → (bs, Q, heads, L, P//D, D, 2)                  4.44 ms
add reference_points                                    14.98 ms
mul 2.0 ; sub 1.0                                       19.09 ms
to_layout ROW_MAJOR                                     14.26 ms   ← 490 M padded elements
                                                                     read to produce 3.8 M real
```
(SCA figures; TSA carried the same chain for another 34.8 ms.)

Every one of those steps is elementwise or a view, and elementwise ops do not care what the axes
mean. So the whole chain now runs on the `(bs, Q, 256)` shape the Linear already emits, in
ROW_MAJOR, where the split into `(heads, levels, points, 2)` is a trailing-axis view rather than a
re-layout:

```
Linear → (bs, Q, 256) TILE
to_layout ROW_MAJOR
add grid_bias                                            ← one add, on real data
reshape → (bs, Q, heads, L, P, 2)                        ← view
```

Two constant folds make that single add sufficient:

- **`2 / [W, H]` in the Linear weight.** `grid == 2·(ref + off) - 1 == (2·ref - 1) + 2·off`, so the
  `2·off` half is a static per-channel scale multiplied into the scale
  [stage 05](05-offset-normalizer-folded.md) already put there. One constant, not two.
- **`2·ref - 1` as `grid_bias`.** The reference half, laid out in the Linear's channel order. The
  Linear emits `(head, level, point, xy)` with the points grouped `(P//D, D)`, and the reference
  point broadcasts over everything but the innermost `(D, 2)` block — so the bias is that flat
  block repeated once per (head, level, point-group), one ROW_MAJOR `repeat`.

The `mul`/`sub`/`to_layout` in `multi_scale_deformable_attn_ttnn` are gone with them; the function
now takes `sampling_grids` already normalized and ROW_MAJOR, and its parameter is renamed to say so.

## Where the time went

| op | stage 06 | stage 07 | Δ |
|---|---:|---:|---:|
| BinaryNgDeviceOperation | 49.3 ms | **2.3 ms** | **−47.0** |
| ReshapeViewDeviceOperation | 77.4 ms | 55.3 ms | **−22.1** |
| UntilizeWithUnpaddingDeviceOperation | 42.4 ms | 26.5 ms | **−15.9** |
| RepeatCodegenDeviceOperation | 1.0 ms | 3.0 ms | +2.0 |
| everything else | 268.3 ms | 269.1 ms | +0.8 |

**`BinaryNg` went 49.3 → 2.3 ms.** That is the whole of candidate 5's opening claim, confirmed: the
arithmetic was never expensive, the padding under it was. Seventeen instances still run; they now
cost 0.14 ms each instead of up to 15.

Op count went **up** by 5 (126 → 131) while kernel dropped 82 ms — two `repeat`s and the extra
`to_layout`s are new. Worth stating plainly: **op count is not the metric.** Candidate 5 was framed
as "reduce the number of ops"; what actually paid was reducing the padded bytes each op touches.

## PCC improved, and that is expected

0.999611 → **0.999651**. Not noise and not luck: the change removes two bfloat16 rounding steps from
every sampling coordinate (`mul` then `sub`, each rounding to bf16) and replaces them with constants
folded at float32 into the weight and the bias. Fewer intermediate roundings, slightly closer to the
torch reference. The change is algebraically exact; only the rounding schedule differs.

## It also fixed the 200×200 OOM

`test_spatial_cross_attention.py::…[nuscenes_base-1-200-200-…]` has failed with a DRAM allocation
error since before stage 04, and both [candidate 1b](../perf_optimization_candidates.md#1b-bound-max_len-statically)
and its [stage-04 re-check](04-fused-msda.md) recorded it as the memory ceiling that made a static
`max_len` bound unaffordable — explicitly attributing it to "the sampling-location math upstream of
the op, which 2 never touched."

That attribution was right, and this change is what touched it. Verified by bisection rather than
assumed: with only [5a](06-sca-key-permute-deleted.md) applied the case still fails; with 5b applied
the full suite reads **33 passed**. The `(bs*Q*heads, L, P, 2)` intermediate that padded to
`32 × 32` was the allocation.

**This reopens 1b.** 1b was rejected on cost (+129 ms of kernel) *and* on the ceiling; the ceiling is
gone, so the entry needs re-pricing against a layer that no longer materializes that tensor. It does
not automatically become a good idea — the +129 ms argument stands on its own — but it can no longer
be dismissed on memory.

## Layer profile now

| Op | inst | ms | % of kernel |
|---|---:|---:|---:|
| MSDAOperation | 5 | 167.9 | 47.1 |
| ReshapeViewDeviceOperation | 21 | 55.3 | 15.5 |
| PermuteDeviceOperation | 10 | 43.7 | 12.3 |
| UntilizeWithUnpaddingDeviceOperation | 18 | 26.5 | 7.4 |
| TilizeWithValPaddingDeviceOperation | 6 | 17.2 | 4.8 |
| SliceDeviceOperation | 13 | 13.2 | 3.7 |
| ScatterDeviceOperation | 1 | 10.5 | 2.9 |
| TransposeDeviceOperation | 12 | 7.9 | 2.2 |
| MatmulDeviceOperation | 11 | 4.7 | 1.3 |
| RepeatCodegenDeviceOperation | 3 | 3.0 | 0.8 |
| BinaryNgDeviceOperation | 17 | 2.3 | 0.6 |

`MSDAOperation` is now **47.1%** of the layer. The upstream questions
([10](../perf_optimization_candidates.md#candidate-10--msdaoperation-itself) /
[12](../perf_optimization_candidates.md#candidate-12--one-fused-call-for-all-levels)) are closing in
on being the only thing left that matters.
