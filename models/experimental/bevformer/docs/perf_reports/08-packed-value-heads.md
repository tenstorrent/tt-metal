# Stage: 08-packed-value-heads

- source commit: [`a63ca3582c7`](https://github.com/tenstorrent/tt-metal/commit/a63ca3582c7)
- harness: `tests/perf/test_layer_perf.py`, one encoder layer, N150, Release build
- kernel time: **91.9 ms** (−20.5 ms), over three runs: 91.9 / 91.9 / 92.3
- op-to-op gap: 63.3 / 41.6 / 34.1 ms over the same three runs — see *What the gap can and cannot say*
- wall: **133.4 ms** median (155.3 / 133.4 / 126.4)
- device ops in the signposted region: **104** (−7)
- PCC gate: **0.999609**, unchanged since stage 06
- CSVs: `generated/profiler/reports/2026_08_28_10_{39_31,41_22,42_28}/`

## What this change was

`multi_scale_deformable_attn` takes a `num_heads` argument. When it is greater
than 1, `value` arrives as `(B, h_in, w_in, num_heads*D)` and the reader picks
head `n % num_heads` out of the stick by byte offset:

```
n_off    = (n / num_heads) * h_in * w_in      page: batch picks it
head_off = (n % num_heads) * D * 2            bytes: head picks a range inside it
```

`D % 16 == 0` was already required, which also keeps `head_off` NoC-aligned.

Nothing else about the op changed: work is still split over `N = B*num_heads`
output tiles, and `grid`, `attn` and the output keep their existing shapes.

## Why this beats a head-reshape op

The obvious fix for 25.9 ms of `Permute` was the deformable member of the
`nlp_create_qkv_heads` family — one kernel doing untilize, channel split, head
permute and level slice in a pass. It would have been the wrong shape.

A fused head-reshape op still has to **produce** the head-major tensor for the
fused op to read: 92.6 MB written, then 92.6 MB read. It removes per-call
overhead, which is not what this costs. The permute ran at 14 GB/s because both
its pages were 64 bytes, not because there were four calls.

Addressing the head by offset means nobody ever produces that tensor. It is the
same move as [stage 07](07-folded-grid-page.md): the axis stops being data and
becomes an address.

## The trade

| op | stage 07 | stage 08 | Δ |
|---|---:|---:|---:|
| PermuteDeviceOperation | 25.86 / 16 | **11.89 / 11** | **−13.97** |
| ReshapeViewDeviceOperation | 17.20 / 14 | **11.89 / 12** | **−5.31** |
| SliceDeviceOperation | 8.86 / 13 | **6.16 / 13** | **−2.70** |
| MSDAOperation | 27.66 / 5 | 28.60 / 5 | +0.94 |

Two ops account for most of it, and they were the two largest in the layer:

- `6x22600x8x32 -> 6x8x22600x32` — the level-0 head permute, **10.01 ms**, gone.
  Its three siblings went with it.
- `180750x256 -> 1446000x32` — the head split feeding it, **4.83 ms**, gone.
  `value` keeps its 512-byte page from the projection all the way down.

The level slice stayed but moved onto that wider page: `6x30125x8x32` at 64 bytes
became `6x30125x256`-shaped work, and the four slices dropped 3.64 → 1.3 ms.

`MSDAOperation` paid 0.94 ms. It now issues the same 64-byte reads at an offset
inside a 512-byte page instead of reading whole 64-byte pages.

By region:

| region | ops | kernel | Δ kernel |
|---|---:|---:|---:|
| SCA — MSDA | 44 | 56.31 ms | **−19.67** |
| TSA — MSDA | 25 | 12.40 ms | **−1.00** |
| SCA — rebatch | 11 | 8.40 ms | −0.11 |
| FFN | 5 | 1.30 ms | +0.01 |

## What the gap can and cannot say

Three runs of identical code gave gaps of **63.3, 41.6 and 34.1 ms** — a 29 ms
spread on a 40 ms quantity. Kernel over the same three runs was 91.9, 91.9, 92.3.

The whole spread sits in one op. `CloneOperation` measures 27–36 ms of op-to-op
latency, and run 1 additionally stalled 20.3 ms on a matmul that did not stall in
runs 2 and 3. That is the `bev_mask` readback — the data-dependent shape of
[candidate 1b](../perf_optimization_candidates.md#1b-bound-max_len-statically) —
and while it is there, wall clock on this harness has roughly ±15 ms of
resolution.

**Two corrections follow from this.** First, the reading offered after run 1 —
that cutting device time makes the device idle longer at the sync — was wrong;
runs 2 and 3 refuted it. It was one noisy sample.

Second, **every wall delta in [PERF.md](../PERF.md) from stage 02 onward is n=1**
and quoted to 0.1 ms. The kernel comparisons are sound — device time reproduces
to a few tenths — but the wall column carries a spread it does not show. Stages
are not being re-measured retroactively; the table now says so.

## Correctness

- `tests/ttnn/unit_tests/operations/experimental/test_multi_scale_deformable_attn.py`
  — **145 passed**. `test_msda_packed_heads` covers `num_heads` 1, 2, 4 against
  `head_dim` 16 and 32, comparing packed value against the head-major permutation
  of the same data; `num_heads = 1` is the legacy path.
  `test_msda_rejects_num_heads_not_dividing_channels` covers the rejected case.
- Full `models/experimental/bevformer/tests/pcc/` — **33 passed**, exit 0,
  nothing deselected.
- `num_heads` defaults to 1, so `models/experimental/vadv2/tt/tt_utils.py` is
  untouched.

## What this changes about the plan

Layout plumbing is **38.4 ms** (was 60.5), still the largest group at 42% of
kernel. The remainder is now concentrated in the attention weights:

| op | ms |
|---|---:|
| `Permute 2496x8x4x4 -> 8x2496x4x4` | 4.99 |
| `Reshape 1x14976x8x16 -> 1x119808x4x4` | 4.46 |
| `Untilize 14976x8x4x4` | 3.61 |
| per-level attn slice ×4 | 2.41 |

That is **15.5 ms of SCA attn prep**, and the trailing `(4, 4)` those shapes
carry pads to a full `(32, 32)` tile — 64× — which is where the untilize goes.
Applying the same offset treatment to `attn` deletes all of it: the op would take
`(B, Q, num_heads*L*P)` and read `P` values at `(h*L*P + l*P)*2` bytes. Grid and
the output follow the same way, for roughly 5 ms and 2 ms.
