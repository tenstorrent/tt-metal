# Exact identity correction specialization

This is an experimental scheduling specialization, not an accepted speedup.
The first E/B resident screens were slower than their frozen v1 controls.

## Guard and numerical proof

UNPACK waits for both maximum CBs and reads the first-column BF16 bit pattern
of every row of both tiles. A group is eligible only if all 64 old/new pairs
are bit-identical and finite. `+0` versus `-0` is deliberately a fallback;
equal signed zeros are eligible. Exponent255 rejects both infinities and NaNs.
The original accurate correction on equal finite maxima produces exact one:
subtraction gives zero, the scale1/sqrt128 multiplication gives zero, and
Blackhole `_sfpu_exp_fp32_accurate_` at zero produces mantissa1/exponent127.
BF16 packing leaves `0x3f80` unchanged.

The independent `identity_probe.py` invokes the frozen subtraction/correction
path for all65,280 finite BF16 encodings. LoFi and HiFi2 device records verify
every first-column result is `0x3f80`, including all subnormals and both zeros.
This does not establish support for nonfinite maxima or arbitrary scales.

## Communication and lifetime

The UNPACK decision is sent through the existing Math/Pack mailboxes. All
three threads take the same branch. The correction CB's reserve, push and pop
remain unchanged even when no correction values are written. Its publication
still drains prior PACK work: the v1 winner removed redundant explicit
PACK_DONE waits relying on this ordering. Removing the token would therefore
be an independent and unjustified synchronization change.

The identity state helper runs only after the applicable DST wait. It loads
exact BF16 one into L6 (numerator) or L6/L7 (denominator), then executes the
same14-instruction arithmetic/round/store bodies as v1. No MAD is rewritten
as an ADD. The skipped instructions are only correction loads. Those bodies
do not overwrite L6/L7; an apparent LOADMACRO argument6 encodes macro1 with
destination L2, not L6. The numerator init reinstalls macro templates and
ADDR_MOD6/7 even when the preceding correction exponentiation did not run.
All canonical DST half clears/releases and state packs are retained.

The finite-bit guard was independently source-reviewed before the first
bounded attention run. Paired K block scales1,1,2,2,4,4,8,8 with distinct V
exercise transitions back into the ordinary correction path. Measured exact
output comparisons include multiple Q jobs and two real trace replays.

## Fast-guard follow-up

`../identity_fastguard/compute_streaming.hpp` changes only guard bookkeeping:
two native-face loops,16 unrolled rows per face, an OR of old/new bit XORs,
and an OR of old exponent255 predicates. Testing the new exponent separately
is redundant: either it differs (already rejected) or it equals finite old.
The complete64-value guard is retained; there is no probabilistic checksum or
sampled-row shortcut. This remains input-dependent and needs separate timing.
