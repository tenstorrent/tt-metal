# Blackhole SFPI → Quasar YAML tests

Scope: the 62 header families classified as Blackhole `SFPI` and Quasar `missing`
in the parity-dashboard snapshot generated 2026-09-10 (commit `3125b4ba98f`).
These are Quasar LLK/fuser tests, not public compute-API or TTNN enablement.

## Use the existing sweep branch

This worktree is based on `marko/fuser-sweep` at
`1b1ac4875571ceb230ba62705d32a252ddb3e657`. Its sweep implementation, loader,
collection plugin, test entry point, generator, allocator, and comparison policy
are unchanged. Our additions register Quasar kernel names, dispatch/init calls,
and independent golden references.

SFPI implementations have been restored to BH source wherever possible, with
documented target/compiler adaptations. The final source passed a fresh full
run of all 278 parity variants and three controls. See
[source-copy details and remaining exceptions](SFPI_BH_SOURCE_PARITY.md).

An operation list means independent test variants, not a fused chain:

```yaml
- type: BinarySfpu
  operation: [SfpuBitwiseAnd, SfpuBitwiseOr, SfpuBitwiseXor]
  src1_dest_tile_index: 0
  src2_dest_tile_index: 1
  dst_dest_tile_index: 2
  iterations: 32
```

The branch expands these choices before normal schema validation, assigning
distinct pytest IDs and generated-kernel names. Each variant uses the same
seeded inputs and the reference belonging to its selected operation.

For another registered kernel with the same node type, input domain, formats,
preparation, and destination layout, append its name to the matching operation
list. A new kernel still needs a dispatch/init registration and an independent
golden reference; the YAML sweep does not supply those automatically.

## Consolidation

The original 278 configurations contained 762 stages. The sweep layout has
168 YAML files that expand to those same 278 configurations and 762 stages:

- 109 original multi-stage files remain byte-for-byte unchanged.
- 169 original single-stage files are represented by 59 templates. Compatible
  configurations differ only in the final SFPU operation and operand naming;
  preparation nodes, constants, formats, indices, and output checks are retained.
- There is at most one operation-list axis per template. No new generator is
  required to maintain or run these files.

The branch forms a Cartesian product across independent list fields. Repeating
a three-operation list in six stages would generate 729 variants, not three.
Operand-format lists also expand independently; they do not pair input/output
formats or change `dest_acc`. Therefore BF16/FP32 and different constant cases
remain separate where needed. We deliberately avoid `operation: all`.

Splitting every original stage into a separate test could reduce the file count
from 168 to 165, but would increase independent compilations from 278 to 762.
The chosen layout keeps the original multi-case tests and their execution cost.

This supersedes the earlier 222-file layout that fused unrelated operation
families into larger pipelines. In that layout, FP32 Erf/Erfc/Erfinv together
reproducibly corrupted Erfc's last vector with half synchronization; full
synchronization passed, but the cause remained unresolved. These tests again
use their original separate configurations. No synchronization workaround or
fuser change is introduced by this sweep migration.

## Running and reviewing

Run from `tt_metal/tt-llk` through the repository run-test skill/runner:

```bash
.claude/scripts/run_test.sh run --arch quasar --test test_fused_quasar.py \
  --k 'sfpi_bh_ and (SfpuBitwiseAnd or SfpuBitwiseOr or SfpuBitwiseXor)' \
  --no-split --maxfail 0 --progress
```

The local emulator wrapper limits each session to 180 seconds. Run bounded
batches of roughly 35–40 expanded variants, not the complete suite in one
session. The branch's unrelated `quasar/sfpu_unary_sweep` example uses `operation: all` and
other sweep axes; do not select it broadly as a small baseline.

- `iterations: 32` covers all four faces of a 32×32 tile; the Quasar dispatcher
  uses eight SFPU iterations per face.
- Ordinary Datacopy (`unpack_to_dest: false`) supports the existing independent
  multi-stage finite-float cases. Direct-unpack integer and exceptional-value
  cases remain single-stage variants, avoiding the existing independent-stage
  semaphore/bank reinitialization limitation.
- Binary cases pack all three tiles, checking the two preserved inputs and
  output. Floating preservation checks use the existing tolerance, not bitwise
  equality. Generic integer-unary references ignore per-tile indices; signed
  binary preparation uses the existing RSUB-based approach where needed.
- Random floating inputs are approximately 0.1–1.1 and random Int32 inputs are
  nonnegative. Explicit constants and signed preparation supply negative cases.
- Floating master checks retain `atol=rtol=0.05`, PCC 0.99; integer master checks
  are exact. These tests do not establish tight ULP parity or exhaust every
  parameter, approximation mode, format, exceptional value, or helper overload.
- Keep the fractional SnakeBeta case: some integer constants do not distinguish
  its small sinusoidal term from a no-op at the default tolerance.
- Cast-rounding discrimination uses direct FP32 input, `CastFp32ToFp16a`, then
  `UnaryEq(0.5)`. These Boolean outputs reject both no-op and truncating casts.

## Coverage boundaries

All 62 targeted families have source implementations and test coverage: 61
directly and `conversions` transitively through power kernels. Fixed-slot
`ParityAddcdiv`, `ParityAddcmul`, `ParityLerp`, `ParitySnakeBeta`, and `ParityMac`
read tiles 0/1/2 and write tile 0. Addcdiv/Addcmul use scalar 0.5. The MAC port
now uses the original Blackhole replay code. Five/six-instruction recordings
were verified in Quasar disassembly and both focused runtime variants passed;
performance parity has not been measured.

`ParityDivInt32Float` transports numbers as Float32, constructs signed integers
in Dest, calls the real integer-input/float-output division helper, restores
input tiles 0/1, and writes the quotient to tile 2. It is not a general
mixed-format fuser extension. Its exact input construction is documented in the
tests/reference.

Structural adapters run once per full tile (`VectorMode::None`). Complex
rotation maps adjacent `(real, imaginary)` columns to `(-imaginary, real)`.
Integer row/column helpers perform partial reductions; tiled product is a
lane-wise prefix product. Independent references model the SFPU vector's gather
of one column parity across four face rows. Alternate mask helpers,
`int_sum::add_int`, and all integer-width/template combinations are not covered.

`SigmoidAppx` uses a Torch quality reference for random inputs.
`SigmoidAppxParity` checks the documented BH three-segment formula at boundaries
and saturation, where its approximation intentionally differs from exact
sigmoid. Neither the comparator nor tolerances have been relaxed.

## Validation

These results validate both the consolidated YAML layout and the subsequent
BH-source-copy cleanup. Its required adaptations and diagnostic evidence are
recorded in [the source-copy notes](SFPI_BH_SOURCE_PARITY.md).

The original 278 configurations passed before this migration. The branch's
actual expansion and schema were used for host preservation checks: exactly
278 variants, all 762 stages, and 762 bit-identical original input tensors
(including NaN payloads). Configuration comparison also verifies that no
preparation, parameter, format, index, or output check changed.

The new layout passed on the Quasar simulator on 2026-09-17: **278/278 variants,
762 stages**, plus three existing fuser control variants. Exact expanded test
IDs were checked against the migration manifest: no missing, duplicate, or
nonpassing cases. The eight parity batches selected 8, 40, 40, 38, 40, 39, 33,
and 40 variants respectively; the separate control batch selected three.

See the [coverage analysis](SFPI_BH_COVERAGE.md) for per-family counts, execution
paths, numerical limitations, uncovered entrypoints, recommended next tests,
and batch evidence. Passing all selected variants does not establish complete
function, branch, or numerical-domain coverage.
