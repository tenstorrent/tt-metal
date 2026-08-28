# Stage: 10-packed-grid

- source commit: [`96aa157b2a6`](https://github.com/tenstorrent/tt-metal/commit/96aa157b2a6)
- harness: `tests/perf/test_layer_perf.py`, one encoder layer, N150, Release build
- kernel time: **69.3 ms** (−4.6 ms), over three runs: 69.3 / 69.4 / 67.9
- wall: **130.0 ms** median (130.0 / 119.0 / 272.5 — the third run is discarded below)
- device ops in the signposted region: **90** (−8)
- PCC gate: **0.999609**, unchanged since stage 06
- CSVs: `generated/profiler/reports/2026_08_28_12_2*/`

## What this change was

The grid joins `value` and `attn`. A rank-3 grid `(B, Q, num_heads*stride*2)`
packs every head and level, and the reader reaches a run at
`(h*stride + point_offset)*4` bytes — four per point rather than attn's two,
since a point is `(x, y)`.

Rank separates it from both existing rank-4 forms, so `(N, Q, 1, P*2)` and
`(N, Q*P, 1, 2)` still work unchanged.

Caller-side that deletes the head-major permute and the four per-level slices:
`sampling_grids` now leaves `TTMSDeformableAttention.forward` as one untilize and
one reshape, and the core attention passes it to all four calls untouched.

## The bug that wedged the chip

`point_offset` was applied to the grid unconditionally. An unpacked grid has
already been sliced to one level by the caller, so its page holds exactly one
run — and offsetting into it reads past the page end. At `num_heads = 1,
num_levels = 4, level = 2` the offset is 32 bytes into a 32-byte page, which
left the clamped read length at zero and the corner indices undefined.

That did not fail a test. It hung the kernel, the broker reaped the job, and the
chip came back **off the PCIe bus** — `MMIO per-op timeout: 4B load took 220636
us`, then `1/1 chip(s) off the bus, below the reset floor`. It took a `tt-smi -r`
from outside this session to recover; the in-band reset path is unavailable on
this host (`recovery ladder unavailable (no systemd)`).

`point_offset` now applies only to an input that is actually packed. The rule the
last three stages converge on: **an offset belongs to the input that packs the
axis, never to the call.**

The same latent bug existed for attn from stage 09 — packed attn happened to be
the only form BEVFormer used, so it never fired. It is fixed with the same
change.

## The trade

| op | stage 09 | stage 10 | Δ |
|---|---:|---:|---:|
| SliceDeviceOperation | 4.25 / 9 | **1.05 / 5** | **−3.20** |
| PermuteDeviceOperation | 6.15 / 9 | **4.34 / 7** | **−1.81** |
| ReshapeViewDeviceOperation | 6.09 / 12 | 5.87 / 10 | −0.22 |
| MSDAOperation | 28.64 / 5 | 29.09 / 5 | +0.46 |

The four `6x8x2496x32 -> 6x8x2496x8` grid slices are gone, and so is the
`6x2496x8x32 -> 6x8x2496x32` head permute. The op pays 0.46 ms for reading a
32-byte block where it read 16.

By region:

| region | ops | kernel | Δ kernel |
|---|---:|---:|---:|
| SCA — MSDA | 33 | 38.41 ms | **−3.97** |
| TSA — MSDA | 22 | 7.74 ms | **−0.95** |
| SCA — rebatch | 11 | 8.49 ms | +0.15 |
| FFN | 5 | 1.30 ms | 0.00 |

## About the third run

Run 3 measured a 204.6 ms gap against 60.7 and 49.6 for the other two, on the
same code, immediately after the chip came back from the reset. Kernel was 67.9
— in line. It is a cold-device artefact, and the wall median is taken over the
three regardless; see [08](08-packed-value-heads.md) for why the gap column on
this harness resolves to about ±15 ms even on a warm device.

## Correctness

- `tests/ttnn/unit_tests/operations/experimental/test_multi_scale_deformable_attn.py`
  — **163 passed**. `test_msda_packed_grid` covers `num_heads` 1 and 4 against
  `(num_levels, level)` of (1,0), (4,0), (4,2) and (4,3). Levels 2 and 3 are both
  the misaligned offsets and, at `num_heads = 1`, the legacy-grid case that hung
  the chip. `test_msda_rejects_packed_grid_point_offset_overrun` covers a run
  that would walk into the next head.
- Full `models/experimental/bevformer/tests/pcc/` — **33 passed**, exit 0,
  nothing deselected.

## What this changes about the plan

Layout plumbing is **15.4 ms**, down from 60.5 at stage 07, and nothing in it is
above 1.9 ms. `MSDAOperation` is **29.1 ms, 42% of kernel**.

One step of [candidate 9](../perf_optimization_candidates.md#candidate-9--axes-as-addresses-not-data)
is left — the output, worth about 1.9 ms, and gated on whether the writer's
destination args honour a byte offset the way the source args do. That question
is still unanswered and, after this stage, is worth answering with a small
device test rather than by reading the header.

What remains after that is not plumbing but real movement: the camera-fold
permute (1.81 ms), the reference-point untilize and reshape (2.9 ms), the value
untilize (1.44 ms), and the attn softmax round trip (2.1 ms across two reshapes).
Roughly 10 ms, none of it addressable by making an axis into an address.
