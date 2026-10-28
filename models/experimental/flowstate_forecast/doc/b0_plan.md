# B0 plan: the fp32 LTI convolution route (CONTRACT v1.1 §6 B0, D0 items 1, 3, 4, 6, 7)

Written during the disk block (D1); nothing here has run yet.

## What is measured

One layer's convolution `x_t = Σ_{k≤t} Lbar^k Bu_{t−k}` alone, in fp32 (state, carry, spectrum and Bu), from Bu to x,
including its layout transforms. Inputs come from the B0 goldens (`tests/golden/make_goldens.py --tier b0`):
layers 0, 3, 4, 5 at

| case | B | L | s | why |
|---|---|---|---|---|
| `s6_L682` | 4 | 682 | 6 | non-aligned L, fastest decay (overflow case for the scaled scan) |
| `s1_L4096` | 2 | 4096 | 1 | the target's length |
| `s0.25_L16384` | 2 | 16384 | 0.25 | the `long_context` guard |
| `s1_12_L16384` | 2 | 16384 | 1/12 | slow modes keep 53-73 % over 16k steps (D0 item 3) |

Accuracy runs on these small batches; timing runs at the target batch (`paper`, 64 x 4096, s = 1) and at
`long_context` (16 x 16384, s = 0.25) on random fp32 Bu of the right shape (timing does not depend on values).
Timings are device time (profiler) or traced time, the median of 5 after warm-up (D0 item 7).

`Bu` is built on the host in fp32 from the bf16 u and the complex64 Bbar (the B-projection matmul is a module test,
not B0). The reference x, Re(C x) and block output are recomputed on the CPU by `b0_reference.py` with the
reference's own expressions and checked against the golden digests; the float64 truth is the direct recurrence
`x_t = Lbar x_{t−1} + Bu_t` in complex128 with Lbar = the reference's complex64 Lbar promoted.

## Routes

**F, FFT.** Bu `(L, B, 512)` → time-last `(B·512, N)` fp32 ROW_MAJOR (real and imag), zero-pad to N = 2048 / 8192 /
32768; `ttnn.experimental.fft(re, im, "precise")`; complex multiply by the cached spectrum
`K̂ = fft(flip(kernel), N)` (host-built per (s, L, layer) from the reference expression, broadcast over B);
`ttnn.experimental.ifft`; slice `[:L]`; back to `(L, B, 512)`. Row counts padded to a power of two
(`two_pass_eligible`, `fft.cpp:360-373`); never the Bluestein path. Expected cost: ~54 GB of DRAM traffic per layer at
`paper` (CONTRACT §1 napkin), i.e. ~0.1 s per layer: likely over the 1.5x-of-dense limit, but the shortest path
to a correct functional model.

**S, chunked complex scan.** c ∈ {32, 64}: intra-chunk `x = Lbar^t ⊙ (Tri @ (Lbar^−t ⊙ Bu))` as one real matmul on
`[re | im]` (HiFi4, fp32 operands at tf32 precision: ~5e-4 relative per term, the main accuracy risk); the modes
where `weight_facts.json` says `|Lbar|^−c · max|Bu|` overflows 1e30 (at s = 6 a handful in layers 4-5) go through a
bounded variant (per-mode Toeplitz blocks `Lbar_p^(t−k)`, |entries| ≤ 1, batched small matmuls with an explicit
program config); carry `h_j = Lbar^c h_{j−1} + x_end` over L/c chunks in fp32 elementwise complex ops; then
`x += Lbar^(t+1) ⊙ h_{j−1}`. Prototype first as a CPU emulation in fp32 and float64 (stage A dataflow), which gives
its error per layer and per channel before any device work.

**K, model-local recurrence kernel (pre-approved, micro-benchmark first).** `ttnn.generic_op`, state in L1 fp32,
time-outer like TiRex-2 `ops/slstm_step`; ~2.1 GB DRAM per layer at `paper`. Built only if F and S both fail the
bound or cost > 1.5x the dense part.

## Bounds and controls (per layer, at L = 682, 4096, 16384)

Items 1-5 of CONTRACT §6 B0: whole-tensor rel-L2 of x (≤ 1e-3, ≤ 1.5e-3 at 16384), per-channel p99 ≤ 1e-2 and max
≤ 5e-2, Re(C x) rel-L2 ≤ 2e-3 and PCC ≥ 0.99999, block PCC ≥ 0.9999, the 2x-reference-error rule for channels where
the reference's own complex64 error against the float64 truth exceeds a bound (listed), no inf / NaN, and route S's
static < 1e30 check over s ∈ {1/12, ..., 6}. Device is measured against the golden; the reference against the
float64 truth (D0 item 6). Route F's margin is reported at every length.

Positive controls that must fail (D0 item 4), run through the same checker: a bf16 state / spectrum, a circular
convolution at N = L, and constants built in float64 instead of the reference's complex64 expressions. A bound whose
control passes is not evidence.

## Decision

Fastest route meeting the bound at every length; within 10 %, stock ops. Written to
`doc/optimize/b0_conv_route.json` (route, ms per layer per shape, items 1-5 per layer, peak memory, git stamp) and a
STATUS line. X gate if no route meets the bound, the best costs > 1.5x the measured dense part of a `paper` layer,
a bound would be loosened, or a precision changes (bf16 Bu included).
