# Log2-lattice maxima: Blackhole feasibility

**Read-only design review; unimplemented and unmeasured on device.** This is
not a change to the selected kernels, an accuracy improvement claim, or a
prediction of throughput. The separate [CPU lattice model](exp_order_lattice_cpu.py)
examines ideal continuous exp, BF16 maximum spills and common-score offsets.
The [completed CPU results](EXP_ORDER_LATTICE_CPU.md) are reported separately;
that model is not an emulation of the proposed hardware instruction sequence.

## Motivation and numerical tradeoff

The continuous Schraudolph surrogate has power-of-two covariance:
`F(x - n*ln(2)) = 2**(-n) * F(x)` for integer n, away from range boundaries.
Lattice anchors and exact power-of-two recurrent rescaling could therefore
remove its dependence on the online maximum history. This does not imply
bitwise permutation invariance of an attention kernel: products, reductions,
packing, underflow and recurrent accumulation remain finite precision.

A fixed absolute lattice introduces a different tradeoff. Arbitrary common
score shifts change the surrogate's mantissa phase. Thus preserving order
consistency can sacrifice arbitrary row-shift invariance. It is not inherently
more accurate than ordinary max subtraction, and common-Q/K stress must not be
replaced by only normal/permutation tests.

## What the current implementation actually does

[The native wrapper](exp_native.hpp) uses the native constants and replay from
[Blackhole ckernel_sfpu_exp.h](../../../tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h):

```
A = FP32(256 * FP32(log2(e)) * FP32(attention_scale))
B = 32500.818359375
code = RNA_sign_magnitude_INT16(A * delta + B)
P = reinterpret(code_magnitude << 15), with sign/underflow handling
```

For D128, A is approximately 32.64446258544922; its score-space unit-log2 step
`256/A` is approximately 7.842065077 and is not exactly BF16-representable.
The MAD itself has hardware rounding, not real-arithmetic semantics.

[fullchip.py](fullchip.py) allocates raw row maxima in BF16 CB10/11, even with
FP32 numerator/denominator/P state. In [compute_streaming.hpp](streaming/compute_streaming.hpp),
`reduce_c_row_group` combines raw current/previous maxima and packs them;
`sub_exp_block_bcast_cols` performs FPU broadcast subtraction before native
exp; `sub_exp_first_col_blocks` subtracts the raw maxima and calls the accurate
correction exponential. The blocked subtraction entrypoint is
[sdpa_sub_custom.h](../../../tt_metal/hw/inc/api/compute/experimental/sdpa_sub_custom.h).

Consequently, simply snapping a real maximum to `ceil(m/step)*step`, spilling
to BF16, and retaining these helpers does **not** establish exact covariance.
The spill, FPU subtraction, coefficient rounding and MAD can perturb the grid.
Selecting a binary-representable score step instead does not solve this unless
its product with the actual A is exactly 256; changing A alone changes the
attention temperature.

## A feasible bounded integer-code prototype

Keep existing raw BF16 maxima and reduction, and derive an integer anchor
`n = ceil(A*m/256)` when a maximum is consumed. For each raw score s:

1. Compute `c = RNA_UINT16(A*s + B)` independently of n.
2. Subtract `256*n` in integer arithmetic, then shift/reinterpret the result.
3. Explicitly handle invalid exponents, underflow and any upper range violation.

For recurrence, load old and new maxima separately, derive their respective
anchors, and construct exactly `2**(n_old-n_new)`. Rounding their difference or
calling the existing accurate exp on the raw difference is not equivalent.
Power-of-two state multiplication is exact only where the representation and
arithmetic admit it; subnormal/underflow behavior remains a test obligation.

This construction makes score-code rounding independent of anchor history.
Keeping raw maxima avoids mixing an integer index with raw scores inside the
existing max-reduction MOP. BF16 maximum error may affect the chosen bound or
positive-logit headroom; it no longer directly changes each score's code phase.
Alternatively, storing integer indices in BF16 requires a bounded exact-integer
range (all integers through magnitude 256 are exact) and a redesigned reduction
that compares indices only after converting each chunk maximum.

The integer-code proposal is deliberately range-limited. For example, scaled
raw logits in [-80,80] give positive pre-round codes approximately 2954–62047,
which fit UINT16. This is an illustrative bounded domain, not a validated
general-input contract. Large common offsets can violate it even when ordinary
max-subtracted softmax would be safe. Generalizing the scheme requires explicit
range handling or a wider conversion design; silently saturating is incorrect.

## ISA traps and scheduling cost

ISA sources were inspected at commit
`5287a62727350bcef35f7b411d1b8a706172ec4c` of
`https://github.com/tenstorrent/tt-isa-documentation`:

- [BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatInt.md](https://github.com/tenstorrent/tt-isa-documentation/blob/5287a62727350bcef35f7b411d1b8a706172ec4c/BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatInt.md):
  current native signed-magnitude INT16 mode 7 saturates at 32767. UINT16 mode 6
  reaches 65535, but **takes absolute value**, not a clamp of negatives to zero.
  Nonpositive pre-round codes therefore need rejection or explicit handling.
  Nearest conversion is ties-away; do not replace it with an RNE CPU oracle.
- [BlackholeA0/TensixTile/TensixCoprocessor/SFPMAD.md](https://github.com/tenstorrent/tt-isa-documentation/blob/5287a62727350bcef35f7b411d1b8a706172ec4c/BlackholeA0/TensixTile/TensixCoprocessor/SFPMAD.md):
  multiplication/addition are partially, not completely, fused; denormals are
  flushed. A bit-perfect model is linked there. LOADMACRO scheduling must
  explicitly respect the MAD dependency gap, and several integer/shift
  dependencies are not automatically detected.

Four FP32 score tiles fill the current half-DST allocation. There is no fifth
tile available for a broadcast maximum. A correct first prototype could use
two score tiles plus the maximum, or a carefully scheduled SFPU-register
resident per-row anchor. The latter must reproduce face/row broadcasts rather
than treating a whole tile's maximum as one scalar. The correction phase also
needs old/new maxima separately, instead of only their FPU difference.

Q/K chunks, KV slots, reader and score/P CB allocation can remain unchanged,
but the score load/subtraction and native LOADMACRO replay need modification.
Extra per-score integer arithmetic, anchor delivery and potentially smaller
exp batches are real costs. Cheaper column-only correction or fewer anchor
changes might offset some cost; neither magnitude nor net throughput is known.
Do not describe this as a free O(rows) max hook.

## Minimum qualification before drawing a conclusion

First qualify scalar/tile code covariance on integer-rounding thresholds,
positive/negative anchors, all admitted range endpoints, underflow and
coefficient/MAD boundary cases against the actual instruction model. Check
per-row/face anchor delivery and original score preservation. Then compare
normal inputs, block permutations and common-score offsets against unchanged
native/LUT controls, preserving original-input references. Require exact
preprocessing, finite outputs, mandatory trace replay and input integrity.
Only a subsequent measured resident/full-chip comparison can establish whether
this tradeoff is preferable to the already measured LUT path.
