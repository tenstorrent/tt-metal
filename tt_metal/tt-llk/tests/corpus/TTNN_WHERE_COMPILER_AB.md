# TTNNWhere compiler-flow A/B status

## Current gate

Do not run the generated selector on silicon yet. Both Blackhole and Wormhole
compile the handwritten and generated Float16_b variants, and CRAQ executes the
handwritten variant correctly, but the generated variant exits CRAQ before
producing either a result or a performance trace.

The test-only selector is in `sfpu_ternary_test.cpp`; neither production
`ckernel_sfpu_where.h` was changed. Selector 0 is the production
SFPLOADMACRO/replay path. Selector 1 expresses the same typed condition and
bit-preserving operand choice with SFPI. Floating-point conditions are compared
numerically, preserving the required `-0 == 0` behavior; selected payload bits
are not converted. All values use the compiler-visible SFPI register API, so
this path has no raw-LREG ownership hole.

## Reproduction

Using compiler SHA256
`f584dce043a26fef1ed8d2d11ef9fb6b4903a61c42b2e9af0c9b0701cca5f360`:

- Blackhole compile producer: handwritten and generated pass (2/2).
- Wormhole compile producer: handwritten and generated pass (2/2).
- CRAQ debug Blackhole at `f2d145ea`: handwritten functional test passes and
  emits a trace; generated exits with rc=1 before a trace or pytest result.
- The CRAQ evidence is under
  `/localdev/nkapre/ttnnwhere-craq-bh-debug` and
  `/localdev/nkapre/ttnnwhere-craq-bh-debug-gen-v2`.

## Compiler finding

The generated straight-line eight-row body is automatically converted to a
14-entry Tensix replay payload. Its dynamic payload is:

1. condition load and condition-to-integer-mask expansion;
2. true and false raw-bit loads;
3. two ANDs, NOT, and OR;
4. bit-preserving store and Dst-counter increment.

The linked `run_kernel` records that 14-entry sequence once and replays it seven
times. This is the first ranked corpus target where generic replay formation
itself creates a simulator-negative binary. Until the exact illegal member or
capture rule is isolated, the generated path is not safe to launch on hardware.
The serial device runner `run_ttnn_where_silicon_ab.sh` is ready but deliberately
not executed.

An earlier, more direct `v_if` spelling also exposed a separate compiler
hardening issue: `rvtt_expand` failed SSA verification for a predicated `vUInt`
selection. The checked-in mask spelling avoids that ICE; it does not explain the
CRAQ exit.
