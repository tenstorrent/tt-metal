# TTNNWhere compiler-flow A/B status

## Update 2026-08-17: condition-mode unification (Lane AG, after the silicon adjudication)

The Lane AD silicon adjudication
(`~/sfpi-uplift/where-adjudication-20260817/verdicts/VERDICT.md`, BH p150,
reset-first, two independent resets) split the WP10 planner-formed shapes:

- separator-kept 4-slot calendar (misc `0x706`, the fp16b/Float32 rows as then
  written): **silicon-RED deterministically** while the byte-identical binaries
  pass CRAQ — a generic-sim delivery mis-model around the TTINCRWC barrier
  (sim owner item), not a device artifact;
- compact 3-slot separator-absorbed calendar (misc `0x770`, Int32 rows):
  **silicon-GREEN in both delivery arms** (RISC-pushed and replay-wrapped).

Per the verdict's remediation path, the generated selector's condition is now
loaded **bitwise** (`DataLayout::U16` for fp16b, `U32` for Float32) and tested
`!= 0` — the handwritten kernel's own protocol (`ckernel_sfpu_where.h` loads
the condition raw LO16/INT32 and predicates on `SFPSETCC LREG_EQ0`). The
earlier paragraph below ("floating conditions are compared numerically,
preserving `-0 == 0`") describes the pre-unification selector and no longer
holds: the unified selector treats `-0.0` as true, exactly as the handwritten
kernel always has. Equivalence proof (exhaustive over all 65536 bf16 condition
patterns + fp32 pattern classes + every suite stimulus class,
`~/sfpi-uplift/laneAG-evidence-20260817/`): `-0.0` is the **only** pattern
distinguishing the two protocols, and no suite stimulus (uniform `[0,1)`,
exact 0/1 patterns) can produce it — golden results are unchanged.

With every load and the store in one launch-sourced mod0, the planner now
derives the compact `0x770` calendar on **all three formats** (dump-verified,
P and R arms), and `-mtt-tensix-macro-planner-replay` wraps the fp16b/Float32
rows too (R != P there now). Verification on the adjudication toolchain pin
(cc1plus `a45dbca55824…`): BH+WH CRAQ 36/36 (both flag states, 9 where nodes,
bit-exact-NaN), and one flocked BH silicon mixed-case PASS per format
(fp16b / Float32 / Int32, planner-ON). Baseline policy per the verdict: the
`where:sem_on` p150 cells book the semantic-OFF-class 251.5 (never formed-ON
numbers) until the compact path books its own cells through the new
impl-parameterized Int32 profile node
(`test_sfpu_ternary.py::test_ttnn_where_int32_device_profile`) and the
now-compact fp16b profile node.

## Current gate (2026-08-15 record, pre-unification)

Blackhole correctness is green for both the production handwritten selector
and the test-only canonical SFPI selector.  The generated selector is not yet a
performance replacement: three fresh physical-device samples measured
`312.50` body cycles versus `159.25` for the handwritten SFPLOADMACRO/replay
path (`+96.23%`).

The test-only selector is in `sfpu_ternary_test.cpp`; neither production
`ckernel_sfpu_where.h` was changed.  Selector 0 is the production
SFPLOADMACRO/replay path.  Selector 1 expresses the same typed condition and
bit-preserving operand choice using canonical `v_if` control flow.  Floating
conditions are compared numerically, preserving `-0 == 0`; selected payloads
use `DataLayout::U16`, which preserves the 16-bit Dst representation and stores
exactly 16 bits.

`DataLayout::LO16` is not interchangeable with `U16`: SFPI documents LO16 as
loading 16 bits but storing 32.  A temporary LO16 spelling therefore produced
periodic zero halves when its selected value had a zero upper half.  That was a
test construction error, not a replay or predicate-lowering failure.

## Compiler fix and gates

The canonical selector originally triggered an `rvtt_expand` SSA verification
ICE under `-O3 -g`.  RVTT replaces scalar comparison results with condition-code
side effects but did not reset `GIMPLE_DEBUG_BIND` uses before clearing or
removing the defining statement.  The SFPI-GCC fix resets those debug-only uses
at both lowering paths.

- SFPI-GCC `nkapre/sfpi`: `8f943c2f8` (the validated fix rebased onto the
  current compiler branch).
- Baseline reproducer: WH, BH, and QSR all fail with `definition in block 2
  follows the use` / `verify_ssa`.
- Fixed compiler: WH, BH, and QSR all compile at `-O3 -g`.
- Non-debug code generation: old and fixed compiler assembly is byte-identical
  on all three reproducers.
- Focused DejaGNU result: 3 expected passes.
- CRAQ Blackhole: generated U16 selector passes with normal compiler replay and
  with `-mno-tt-tensix-optimize-replay`.

## Blackhole silicon evidence

The serialized run used the source change later rebased and pushed as TT-Metal
`1dd59d9c`,
compiler SHA256 `569fb8fd0f0a267dd566cdc740a577c76ef69cdeea905a64e7c79e6b1f19ebfd`,
and host `bh-33-special-nkapre-for-reservation-112326`.

| implementation | correctness | body cycles r1/r2/r3 | `.text` SHA256 |
| --- | --- | --- | --- |
| handwritten macro/replay | pass | 159.25 / 159.25 / 159.25 | `c203a14707dc8514fbf41048385a1ea5816221fe354f76fe2970adeb26db873c` |
| generated canonical SFPI | pass | 312.50 / 312.50 / 312.50 | `616263dcfb2caefbaeb6eb1692be41abf78acc8e0b10812a8929c6adcf1d29d0` |

Raw and post-processed profiler CSVs, per-run math ELFs and disassemblies,
ELF/text hashes, compiler provenance, `tt-smi`, logs, and a verified manifest
are archived at:

`/localdev/nkapre/ttnn-where-bh-silicon-debugbind-u16-20260815`

Full ELF hashes differ between repetitions because debug metadata embeds the
per-run temporary path; executable `.text` hashes are stable within each A/B
arm.

The hot generated body already receives the general outermost-condition-code
combine: its replay payload is seven instructions (`SFPLOAD` condition, true,
and false; `SFPSETCC`; predicated `SFPMOV`; `SFPENCC`; `SFPSTORE`), with
`TTINCRWC` outside the payload.  There is no `SFPPUSHC`/`SFPPOPC` pair in that
executed body.  The handwritten payload is three instructions: two
`SFPLOADMACRO` words (`0x9306e000`, `0x9386e040`) followed by the false-value
`SFPLOAD` (`0x7006c080`).  It therefore uses 3 replay slots per face versus 7
for generated SFPI.  The generated ELF also retains an unused out-of-line
always-inline wrapper containing PUSH/POP; it is not the profiled path.

## Decision

Correctness/compiler gate: **GO-BH-ONLY**.  Performance replacement gate:
**NO-GO**.  Closing the performance gap requires a general compiler mechanism
that can form the hardware SFPLOADMACRO pipeline (the F1 track), rather than a
Welford/where-specific peephole or a production handwritten-kernel edit.  A
new PUSH/POP-to-ENCC optimization is not the missing win: `rvtt_cc` already
applies that guarded transform only at outermost CC-stack depth, while nested
predicates retain stack semantics.  A focused future regression should inspect
the instantiated caller's function body (not the whole assembly, which also
contains skipped always-inline wrappers) and assert SETCC/MOV/ENCC with no
PUSH/POP.
