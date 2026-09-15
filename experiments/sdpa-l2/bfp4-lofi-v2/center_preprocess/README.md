# Fused FP32 SFPU K centering and quantization

Isolated prototype; no previously qualified packer is modified or imported.

`build(device, src, bias, ncores=1, mode="b8_rne5")` returns
`(output, invoke, actual_cores)`. Both inputs are BF16 TILE/DRAM tensors:
`src=[1,H,N,128]`, `bias=[1,H,32,128]`, with all 32 bias rows repeating the
per-head column bias. `N` is a positive multiple of 32. The output is BFP8 for
`b8_rne5`, or BFP4 for `b4_rne`. The initial prototype uses BF16 DST, batch4.

Each core owns a contiguous, four-tile-aligned segment, possibly crossing head
boundaries. CB1 holds four bias tiles for the current head; it is reloaded only
at a boundary. Its single-buffer reserve/pop protocol prevents overwriting
live bias. Input and output CBs are double-buffered. Each batch copies four
input and four bias tiles into the eight BF16 DST slots. SFPU loads the paired
tiles, subtracts in FP32, and immediately quantizes without spilling the
centered value through BF16. Only exactly BF16-representable quantized values
are stored back before native packing. Bias is cached in L1, not assumed to
survive DST ping-pong. Neither FPU ELWSUB nor input-format switching is used.

## Numerical contract

Inputs and bias must be finite BF16 normals or zero. The FP32-centered result
must be zero or an FP32 normal of magnitude at least `2^-126`; each nonzero
native 16-column group maximum must have exponent in `[-124,106]`. Arbitrary
subnormal/overflow behavior is outside scope. The CLI checks this before opening
the device; the reusable build API does not download inputs to validate them.

The reference subtraction is `src.float() - bias[:, :, :1, :].float()`, not
a BF16-centered tensor. BFP4 uses two separate FP32 magic additions and
saturation, cross-checked against direct FP64 grid RNE with all FP32 input bits.
The older BF16-only aligned-integer host oracle is deliberately not reused:
discarding alignment sticky bits can change a general FP32 tie decision.
BFP8 uses per-value integer RNE5 then native shared-exponent nearest-away
rounding, not host BFP8 ties-even packing.

An intermediate BF16 spill is not generally equivalent. For example, BF16
`K=1.03125`, `bias=-2^-10` produce FP32 `1.0322265625`: direct RNE5 yields
`1.0625`, while a BF16 spill rounds to `1.03125` and then RNE5 yields `1.0`.
For `K=1.125` with the same bias and native group exponent zero, direct BFP4
gives `1.25`, versus `1.0` after a BF16 spill. These are threshold examples,
not claims about aggregate normal-distribution error. The CLI reports the
actual spill-induced mismatches and output L2 for each input. For large
same-sign common mode, Sterbenz exact subtraction often makes the centered
result exactly BF16 already, eliminating this particular difference.

## Smoke commands

Only the card owner should run these. H3/cores4 forces head-boundary crossings
and segments that start inside a head; H1/cores1 alone does not cover caching.

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/center_preprocess.py --label center8-normal-v1 --mode b8_rne5 --heads 3 --cores 4
python experiments/sdpa-l2/bfp4-lofi-v2/center_preprocess.py --label center4-normal-v1 --mode b4_rne --heads 3 --cores 4
python experiments/sdpa-l2/bfp4-lofi-v2/center_preprocess.py --label center8-common-v1 --distribution common_k
python experiments/sdpa-l2/bfp4-lofi-v2/center_preprocess.py --label center4-wide-v1 --mode b4_rne --distribution wide
python experiments/sdpa-l2/bfp4-lofi-v2/center_preprocess.py --label center4-ties-v1 --mode b4_rne --distribution thresholds
python experiments/sdpa-l2/bfp4-lofi-v2/center_preprocess.py --label center8-perf-v1 --heads 10 --length 32768 --cores 110 --iters 10
```

Also test zeros, H1/cores1, and the other quantization mode for each distribution.
`--host-only` avoids TTNN import and device access. Mean generation is currently
CPU-based and excluded from the device timing; a later TTNN mean can supply the
same bias tensor. The added static work is one bias tile copy plus two SFPU
loads, two FP32 subtract instructions and one scheduling NOP per 64 values
(80 SFPU instructions per tile), plus 8192 bias bytes per core/head segment.
No device performance claim is made before measurement.
