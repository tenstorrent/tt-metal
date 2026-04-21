# W-Seam Minimal Test Plan

## Goal
Build the smallest possible TT-vs-torch diff that exposes the W-boundary mosaic artifact seen in 2x4 480p WAN T2V VAE decoder output, so we can iterate on the remaining hypotheses without running the full decoder.

## What we know

### Observed
- 3 vertical mosaic lines at W-device boundaries of the final decoder video, 2x4 mesh on Loud Box P150b.
- Mesh axes: `axis 0 = H (2)`, `axis 1 = W (4)`.
- Chip layout (from zero-pad distribution in Diagnostic A/B): col0=`{0,1}`, col1=`{4,5}`, col2=`{6,7}`, col3=`{2,3}`.

### Ruled out (with evidence)
| Hypothesis | Evidence |
|---|---|
| W-writer / conv3d-reader addressing mismatch | Both compute `t*h_total*W_dev + h_padded*W_dev + w_col` identically. |
| Phase-2 W-reader skips H-top corner | `WR_HC1=0` is correct: boundary chips take `is_padding_zeros` branch before reaching the read. |
| Ping-pong halo buffer staleness | Diagnostic B: 0x`deaddead` sentinel hits = 0 → every page is written by the current dispatch. |
| W-writer coverage bug | Same sentinel result: writer touches every page it owns. |
| Phase-2 progress-sem race (shared sem across directions) | Each direction's W-reader sits on its own CoreCoord with its own sem count. |

### Not yet ruled out (priority order)
1. **Conv3d reader halo-addressing at `w_in=-1` / `w_in=W_dev`.** Halo buffer content is correct; the reader may be picking the wrong `(t, h)` row. Bug lives in `gather_rows_halo` inside `reader_vol2col.cpp`.
2. **Final stitching / gather / reshard** on the 4 W-slabs after the decoder. A seam off-by-one here is indistinguishable at the video level.
3. **Compute-side boundary handling**: reducer/worker split or fidelity differs for boundary sticks.
4. **Artifact not in the fused NP+Conv3d path at all** — could be in `WanConv2d` or a standalone NP op.

## Minimal test design

### File
`models/tt_dit/tests/models/wan2_2/test_w_seam_minimal.py`

### Op path tested
`ParallelManager.neighbor_pad_conv3d_fused(...)` only (same code path WAN uses in `v5`). Inherits all barrier / progress-sem / halo-buffer plumbing from `manager.py`, so no boilerplate duplication.

### Shape
```
per-chip: [B=1, T=1, H=4, W=4, C=4]
full mesh: [B=1, T=1, H=8, W=16, C=4]
```

### Kernels and inputs

Three scans, run in order. First is a visual sanity check; scans L and R pinpoint which halo direction is broken.

**NOTE (2026-04-18):** `(1,1,3)` kernel does **not** exercise the fused NP+Conv3d path as intended — `WanConv2d` only emits `dims=[3]` (W-only), but the fused op assumes `dims[0]=H, dims[1]=W` and reinterprets the single pad as H-padding applied along the W mesh axis. Result: per-chip output is `H+2 × W-(kW-1)` instead of `H × W`, and no W halo is delivered. We use a `(1,3,3)` kernel so both H and W halo branches fire.

| Scan | kernel | weights | input | output formula | exposes |
|---|---|---|---|---|---|
| **U** (uniform) | (1,3,3), pad=(0,1,1) | all `1.0` | all `1.0` | `out = C_in * (H_taps) * (W_taps)` where `taps = 2` at global edge else `3` | any W-seam at a glance |
| L | (1,3,3), pad=(0,1,1) | `[1,0,0]` along W, all `1.0` along H (center row only) | `in[b,t,h,w,c] = global_w + 0.01*h + 0.001*c + 1` | `out[h,w] = sum_kh in[h+kh-1, w-1]` | which `(h,w)` left halo reads |
| R | (1,3,3), pad=(0,1,1) | `[0,0,1]` along W, center row only | same as L | `out[h,w] = sum_kh in[h+kh-1, w+1]` | which `(h,w)` right halo reads |

Bias = 0. Stride = 1. Dilation = 1. `C_in = C_out = 4`.

#### Scan U expected output

With `C_in = 32` and `(1,3,3)` kernel (9 taps), each output value = `32 * (# valid taps)` where a tap is valid iff it falls inside the global tensor (no zero-pad):

```
interior  (not on any global edge): 32 * 3 * 3 = 288
edge      (on exactly one global edge): 32 * 2 * 3 = 192
corner    (global (h=0|H-1, w=0|W-1)):  32 * 2 * 2 = 128
```

Only the outermost row / col of the full 8x16 global output should drop below 288.

A broken W-halo shows as an interior W column of a middle-col chip collapsing from `288→192` (on interior H rows) or `192→128` (on global H borders). Instantly visible when per-chip grids are printed side-by-side.

#### Scan L/R rationale (only run if U fails)
- Scan U tells you a seam exists. Scans L and R pinpoint which halo direction is wrong and at which `(h, w)` — see failure table below.
- The coordinate-encoded input makes every mismatched value self-describing: a wrong value immediately tells you which `(h, w)` position the reader actually pulled.

### Expected outputs — scan L (left-only)
Let `W_start` be the chip's absolute W offset in the global tensor. For every `(b, t, h, w, c)` on every chip:
- If `w == 0` and chip is in col 0 (`{0,1}`): `out = 0` (zero-pad, no left neighbor).
- Otherwise: `out = in_global[b, t, h, W_start + w - 1, c]`.

Symmetric for the right-only scan at `w = W_dev - 1`.

### Failure modes this test catches
| Symptom in output | Implied bug |
|---|---|
| Halo position reads 0 on a middle-column chip | Halo row not consumed; reader addressing off. |
| Halo position reads right numeric value but wrong `h` | Conv3d reader uses wrong row index from halo buffer. |
| Halo position reads a value off by exactly 1 `w` | Off-by-one in halo `w_col` selection. |
| Col-0 chip produces nonzero at `w=0` | Zero-pad path not taken. |
| All values correct | Bug is outside the fused NP+Conv3d path; pivot to stitching / standalone NP / WanConv2d. |

### Diff report
Per-chip grouping, print only rows with any error:
```
chip=4 col=1 side=LEFT
  h=0 c=0:  tt=4.001   ref=4.001   OK
  h=1 c=0:  tt=0.000   ref=4.011   BAD (zero)
  h=2 c=0:  tt=4.021   ref=4.121   BAD (wrong h: read h=2 instead of h=12??)
  ...
chip=5 col=1 side=LEFT
  ...
```

### Success / failure thresholds
- `max_abs_err < 1e-3` → test passes.
- Any mismatch → test fails with the diff report above.

## State
- [x] Hypothesis list consolidated, doc drafted.
- [ ] Revert `get_np_halo_buffer` sentinel back to `torch.zeros(...)` before running this test (sentinel poisons output).
- [ ] Write `test_w_seam_minimal.py` with scans U, L, R.
- [ ] Run scan U; visually inspect per-chip output map for interior `8`s.
- [ ] If U passes, skip L/R and pivot to stitching / standalone NP / WanConv2d.
- [ ] If U fails, run L and R; classify failure mode from the table above.
- [ ] If it passes, pivot to stitching / standalone NP / WanConv2d.
- [ ] If it fails, use the failure mode to locate the bug in `reader_vol2col.cpp`.

## Decisions
| Decision | Reason | Rejected Alternative |
|---|---|---|
| Use fused path only | Matches what WAN uses in `v5`; halves surface area. | Also test standalone NP + standalone conv3d |
| Kernel (1,1,3) | Isolates W-axis halo path; no H/T halo confounds. | 3x3x3 (harder to reason about) |
| Run uniform-all-ones scan first | Output collapses to `{8, 12}`; a broken seam is visible at a glance with no tooling. | Skip and go straight to one-hot |
| One-hot weights for L/R | Output is literally the halo value; no averaging to obscure a bug. | Uniform box |
| Additive coord input for L/R | Any wrong index shows as a numerically explainable diff. | Random input (hard to trace root cause from value alone) |

## Open questions
- [ ] Does the artifact need `T > 1` to reproduce? (Decoder uses multiple frames; minimal test starts at `T=1` — expand if test passes but real decoder still fails.)
- [ ] Does the artifact need multi-channel C > 4? Unlikely but verify if `C=4` passes and real decoder still fails.
