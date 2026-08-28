# Stage: 09-packed-attn-runs

- source commit: [`978fc933566`](https://github.com/tenstorrent/tt-metal/commit/978fc933566)
- harness: `tests/perf/test_layer_perf.py`, one encoder layer, N150, Release build
- kernel time: **74.0 ms** (−17.9 ms), over three runs: 73.9 / 74.4 / 74.0
- op-to-op gap: 56.1 / 45.2 / 59.9 ms — see [08](08-packed-value-heads.md) *What the gap can
  and cannot say*
- wall: **130.0 ms** median (130.0 / 119.6 / 133.9)
- device ops in the signposted region: **98** (−6)
- PCC gate: **0.999609**, unchanged since stage 06
- CSVs: `generated/profiler/reports/2026_08_28_11_3{4_48,5_18,5_50}/`

## What this change was

`attn` may now arrive as `(B, Q, num_heads*stride)` — every head's every level in
one row. Two new arguments name the run this call wants:

- `num_points`: the points sampled per query
- `point_offset`: how far into a head's run they start

The reader reaches head `h`'s level run at `(h*stride + point_offset)*2` bytes of
the query's row. The caller stops permuting `attn` head-major and stops spelling
out `(num_levels, num_points)`.

Deleting that spelled-out form is most of the win. `(4, 4)` as trailing dims pads
to a whole `32x32` tile — **64×** — which is what a 3.6 ms untilize was paying
for, and what a 4.46 ms tiled reshape was building.

## Two things this got wrong first

**A NoC source offset must be 32-byte aligned.** The first version passed
`(h*stride + point_offset)*2` straight to `async_read` and failed PCC on exactly
the cases where that is not a multiple of 32:

| offset | 0 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|
| result | ok | **fail** | **fail** | **fail** | ok | **fail** | ok | ok |

Stage 08's value offset is `h*D*2` with `D % 16 == 0`, so it is always 32-aligned
and the constraint never showed. attn's is `(h*L*P + l*P)*2`; at BEVFormer's
`P = 4, L = 4` levels 1–3 give 8, 16 and 24. The reader now rounds the read down
to the boundary, indexes the wanted points from there, carries the worst-case
30-byte lead-in in its scratch row, and clamps the read at the page end.

**`num_heads == 1` makes `N` and `B` the same number**, so the batch dimension
cannot tell packed attn from legacy `(N, Q, P)`. The row width can: attn is
packed exactly when its last dim differs from `P`.

**A claim from stage 08 was also wrong.** [Candidate
9](../perf_optimization_candidates.md#candidate-9--axes-as-addresses-not-data)
said each input could be widened independently. `attn` and `grid` can — they are
selected by their own shapes — but `value` cannot: `num_heads > 1` *is* the
switch that makes value packed. Both are now selected the way that actually
holds, and the candidate says so.

## The trade

| op | stage 08 | stage 09 | Δ |
|---|---:|---:|---:|
| PermuteDeviceOperation | 11.88 / 11 | **6.14 / 9** | **−5.75** |
| ReshapeViewDeviceOperation | 11.86 / 12 | **6.30 / 12** | **−5.55** |
| UntilizeWithUnpadding | 8.56 / 9 | **4.11 / 8** | **−4.45** |
| SliceDeviceOperation | 6.14 / 13 | **4.43 / 9** | **−1.71** |
| MSDAOperation | 28.62 / 5 | 28.64 / 5 | +0.02 |

The four SCA attn ops that went: `Permute 2496x8x4x4 -> 8x2496x4x4` (4.99),
`Reshape 1x14976x8x16 -> 1x119808x4x4` (4.46), `Untilize 14976x8x4x4` (3.61) and
the four level slices (2.41). A 0.80 ms reshape came back — the head axis is
still spelled out for the softmax, which normalises over each head's run, and
folded again straight after.

The op paid nothing. Its attn read is 32 bytes where it was 8, but it was one
transaction either way and attn is 3.8 MB against value's 92.

By region:

| region | ops | kernel | Δ kernel |
|---|---:|---:|---:|
| SCA — MSDA | 39 | 42.38 ms | **−13.93** |
| TSA — MSDA | 24 | 8.69 ms | **−3.71** |
| SCA — rebatch | 11 | 8.34 ms | −0.06 |
| FFN | 5 | 1.30 ms | 0.00 |

## Correctness

- `tests/ttnn/unit_tests/operations/experimental/test_multi_scale_deformable_attn.py`
  — **154 passed**. `test_msda_packed_attn` covers `num_heads` 1 and 4 against
  `(num_levels, level)` of (1,0), (4,0), (4,2) and (4,3) — the last two are the
  misaligned offsets — comparing the packed row against the head-major slice of
  the same data. `test_msda_rejects_point_offset_past_the_head_run` covers a run
  that would walk into the next head.
- Full `models/experimental/bevformer/tests/pcc/` — **33 passed**, exit 0,
  nothing deselected.

## What this changes about the plan

Layout plumbing is **20.6 ms**, down from 60.5 two stages ago, and no single item
is above 1.9 ms. `MSDAOperation` is **28.6 ms, 39% of kernel** — the largest
item in the layer again, for the first time since stage 06.

What is left of the layout work, and what it is worth:

| item | ms | what it needs |
|---|---:|---|
| grid head permute + level slices | ~4.5 | the same treatment, [candidate 9](../perf_optimization_candidates.md#candidate-9--axes-as-addresses-not-data) |
| output concat heads (permute + reshape) | ~1.9 | writer-side offset, unverified — see below |
| camera fold permute, reference-point prep | ~6.5 | real data movement, outside the op |
| value untilize, attn softmax round trip | ~3.3 | real |

Still unverified: `writer_msda.cpp:87` passes `{.offset_bytes = 0}` in its
*destination* args. Whether that struct honours an offset the way the source
struct does has to be checked before the output step — the alignment lesson above
suggests checking it on the device rather than by reading the header.
