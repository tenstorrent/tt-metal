# Binary-broadcast compiler-flow A/B

## Result

**PARITY — no regression and no speedup.**  The test-only generated SFPI
arithmetic matches the handwritten COL/ADD body at exactly 608 Blackhole
device cycles in all three fresh samples.  This is a successful conversion
gate, not a performance win.

Production LLKs are unchanged.  The selector and generated implementation live
only in `tests/sources/sfpu_binary_bcast_test.cpp`; replay, transpose, address
modifiers, loads and stores remain the existing architectural operations.  GCC
owns the ADD/SUB/MUL arithmetic islands through typed `sfpi::vFloat` values and
fixed `sfpi::l_reg[]` endpoints.

## Evidence

- TT-Metal source: `e0128f3bdbc34014a62e6c445ea97e60909a7dac`
- Compiler SHA-256:
  `1164fde97fbba70ba7aa279b104f38a0b252a9bd8ed0fbe86b2e4858e95a2a4c`
- Full archive:
  `/localdev/nkapre/binary-bcast-bh-silicon-20260815`
- Archive manifest: `SHA256SUMS` in that directory
- WH compile-only generated matrix: 6/6 (ROW/COL x ADD/SUB/MUL)
- BH representative correctness matrix: 8/8 including both handwritten
  baselines and all six generated operation/dimension combinations
- Serialized archive correctness gate: 2/2 for identical COL/ADD inputs

| implementation | BH body cycles | stable `.text` SHA-256 |
|---|---:|---|
| handwritten replay | 608, 608, 608 | `4ebbc0a97f7ef4f911fc9364953d979d5f5759db7e7a4bc5d986eaee5eaf9625` |
| generated SFPI | 608, 608, 608 | `5b87bcc2fb6ca9083b095975035dd188e6d2aa7dc8a7eca4f5aee1923de7fd63` |

The binaries retain the same replay structure: one record plus eight
`TTREPLAY` executions (`9` static `ttreplay` instructions in the full math
image) and the same 14 static `sfpnop` instructions.  The generated image has
32 decoded `sfpadd` instructions, four per row band.  The corresponding 32
handwritten TTI encodings are bit-different and are not decoded as `sfpadd` by
this objdump, but occupy the same locations and produce the same silicon time.
Apart from those arithmetic encodings, the relevant code differs only by one
independent scalar instruction ordering change.

## CRAQ gate

CRAQ functional execution passes both selectors.  The combined simulator was
the cycle-restored `f2d145ea` source plus the BH debug reset-PC readback needed
by the transport; its `libttsim.so` SHA-256 is
`6c03401dee15ca74e7dab0e6c13cf60928314197a12cb17b1e858c68262d138a`.
Evidence is in `/localdev/nkapre/binary-bcast-craq-bh-fixed`.

Both rows are identical in the current modeled aggregate: 144 launches, 3421
simulated cycles, 1.79167 normalized SFPU instructions and 4.13194 normalized
total stalls.  These are simulator/debug features, not device-cycle evidence;
the 608-cycle scoped silicon markers above are the performance authority.

The first CRAQ attempt used a cycle-restored binary without the reset-PC
readback and failed before kernel launch with no trace.  That was simulator
plumbing, not a binary-broadcast correctness failure.

## Compiler implication

Typed SFPI can replace the broadcast kernel's four-op arithmetic island without
cost.  It cannot improve this already hand-pipelined path while replay,
transpose and dynamic addressing remain opaque architectural barriers.  A
general future win requires compiler ownership or explicit typed modeling of
those operations so replay formation and scheduling can see a larger region;
special-casing binary broadcast or deleting the existing replay would not be a
valid compiler improvement.
