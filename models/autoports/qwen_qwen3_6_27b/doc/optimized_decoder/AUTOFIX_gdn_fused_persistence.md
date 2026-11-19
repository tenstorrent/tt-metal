# Fuse persistence rounding into update/readout

The packed prepared-Q/K helper passed 54 bit-exact cases and measured 1.765976 ms
B32 short decode, versus 1.809059 ms for expanded Q/K. This candidate builds on
that helper and targets the remaining separate state-persistence traversal.

The incumbent stores each updated FP32 state value, computes the current Q
readout from the unrounded live value, then executes the native FP32-to-BF16
typecast over all 16 state tiles. The candidate copies each live updated LREG
into scratch and performs the native integer RNE sequence before its state store.
The original DATA register remains unrounded for the Q product and accumulation.
The later full-state typecast pass is removed. FP32 persistent-state mode keeps
the original direct store.

The exact native expression is `((bits + 0x7fff + ((bits >> 16) & 1)) &
0xffffffff) & 0xffff0000`. The implementation copies the native shift, two masks
and two adds from `calculate_typecast_fp32_to_fp16b` in
`ckernel_sfpu_typecast.h`. It uses LREG2 and LREG6 as scratch; packed Q/K direct
loads no longer consume the column mask. Native typecast init installs constants
12/13/14 before each update traversal. Column init then restores zero-stride
MOD7 without changing those constants. The state store uses FP32 mode, exactly
as the native typecast does. No hardware stochastic-round instruction is used.

This removes the unrounded intermediate state store and subsequent typecast
reload/store traversal, replacing the reload with an LREG copy. The net static
instruction saving is modest; reduced DST traffic and traversal overhead are
the performance hypothesis. No latency gain is claimed. Reader, writer, CBs,
partition, matrix arithmetic order, reduction tree and final pack are unchanged.

Frozen files:

- `candidates/gdn_sfpu_fused_persistence.py`
- `candidates/integrated_fused_persistence134.py`
- `candidates/make_gdn_fused_persistence_candidate.py`
- `candidates/check_gdn_fused_persistence_cpu.py`
- `candidates/gdn_fused_persistence_cpu.json`

CPU/source checks passed. They compare all 65,536 BF16 high words against eight
low-word boundary patterns (524,288 cases), prove identical native integer RNE
bits and untouched live values, and check unchanged recurrence arithmetic and
state pack. Python compilation passed. Device compilation, register scheduling,
exactness and latency remain unverified. No device or main/shared source was
touched.

Parent validation: existing `run_recurrence_exact_comparison.py`, candidate
`gdn_sfpu_fused_persistence.py`, reference `gdn_sfpu_packed_prepared_qk.py`,
`--state-dtypes bf16 fp32`. Require 54 zero-mismatch cases, then whole B32
short/512 and B1 short; profile the recurrence if the gain exceeds run noise.
