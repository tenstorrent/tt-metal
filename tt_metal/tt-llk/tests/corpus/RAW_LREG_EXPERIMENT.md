# Raw LREG lifetime experiment — 2026-10-07

## Current conclusion

Existing read/write builtins can express a working explicit-state alternative
for the tested regions, even with the live-in pass disabled. Independent pairs
alone are insufficient: preserve input lifetimes (including inactive destination
lanes), retain output clobbers, and carry live outputs to their consumers.

The complete current O2 device suite finished **370 PASS / 14 diagnostic XFAIL**,
zero unexpected failures, 384 cases. The broader and focused matrices below
establish additional configuration coverage, not universal compiler correctness.
No global production-wrapper rewrite or compiler-pass removal is justified.
The final section records the scoped, opt-in Blackhole production entry point.

The production-linked TopK merge experiment below additionally passes exact
value/index checks with the pass disabled. Its default-codegen performance
regression is associated with an unroll-cost threshold, not extra vector moves.
It does not justify global pass removal or changing production defaults.

## Question and controlled intervention

Do independent read/write identity pairs preserve a raw L0 value across a typed
SFPI temporary? Does explicitly threading a C++ value change the result?

The initial `sources/sfpu_raw_lreg_device.cpp` produced 1.0 in raw L0, loaded/stored a typed
2.0 temporary, then stored raw L0 to the output. The independent oracle was 1.0
in every output element; the input tile had to remain 2.0. Both TT (MMIO) and TTI
instruction forms run. Only the annotation scheme changes:

0. No annotations (diagnostic control).
1. `writelreg(readlreg(0), 0)` after producer and before consumer.
2. Raw effect metadata after producer and consumer.
3. `saved = readlreg(0)` after producer; `writelreg(saved, 0)` before consumer.

Scheme 3 is an explicit use-def dependency. It is **not** an implementation of
Nathan's proposed `sfpvalue` builtin, and does not test that proposed API.

## Initial measured results

Physical Blackhole on `ttuser@tt-quietbox-0.local`, runtime checkout
`b8c16a2977d1674c49980f1802097e5205ac5187` plus these test files;
standalone PR #22 compiler `bc27e710ef47be5ac403476230dd8b2e8db8255a`.

| Scheme | O2 TT wrong elements | O2 TTI wrong elements |
|---|---:|---:|
| No annotation | 1024/1024 | 1024/1024 |
| Independent pairs | 1024/1024 | 1024/1024 |
| Effect metadata | 0/1024 | 0/1024 |
| Threaded value | 0/1024 | 0/1024 |

Additionally, scheme 3 passed with `-fdisable-rtl-rvtt_lreg_livein` at O2 and O3,
each with default scheduling and with `-fschedule-insns -fschedule-insns2`:
eight hardware cases, zero wrong elements. Thus this explicit region does not
need the new live-in pass. The standalone O2 scheduled TTI assembly with the
pass disabled has READ L0, temporary SFPLOAD/SFPSTORE L1, WRITE L0, and no extra
move. This is an instruction observation, **not** a latency measurement.

O0 hardware validation did not run: the shared BRISC harness fails compilation
in `t6_semaphore_init`, with an impossible immediate constraint for TTI_SEMINIT.
Do not count that as a pass or as a device failure of scheme 3.

## Re-run

Use the matching SFPI compiler/headers with `sfprawlreg_effect` support. From
`tt_metal/tt-llk/tests/python_tests`, on a Blackhole device with the LLK test
environment installed:

```sh
CHIP_ARCH=blackhole TT_LLK_EXTRA_COMPILER_OPTIONS='-O2' \
  ../.venv/bin/python -m pytest test_raw_lreg_device.py -s -q

for opt in O2 O3; do
  for scheduling in '' '-fschedule-insns -fschedule-insns2'; do
    CHIP_ARCH=blackhole \
    TT_LLK_EXTRA_COMPILER_OPTIONS="-$opt $scheduling -fdisable-rtl-rvtt_lreg_livein" \
      ../.venv/bin/python -m pytest -s -q test_raw_lreg_device.py \
      -k threaded || exit
  done
done
```

For a separately built compiler, prepend `-B/path/to/compiler/backend/` and
`-I/path/to/matching/sfpi/include` to the options, as in this experiment.
Do not use `--compile-producer`: that does not execute the hardware assertions.
XFAIL means wrong output was observed in an intentionally incomplete diagnostic
scheme, not correctness success. Effect and complete threaded cases must pass;
discarded reads and output-only threading across an old-input gap are controls.

The companion `raw_lreg_full_annotation.cpp` is an assembly-only comparator;
`selftest_raw_lreg_metadata.sh --target-cxx /path/to/riscv-tt-elf-g++` exercises
all five straight-line schemes plus the pressure matrix. Allocation scans alone are not a correctness oracle because
a compiler may legally preserve a value with moves.

## Decision and remaining scope

Independent identity pairs are insufficient in this experiment. Explicit
threading is a working alternative for a producer/consumer region whose value
can be carried in C++. This does not establish that independent macro wrappers
can communicate that lifetime, nor that the effect pass can be removed globally.

No production wrapper policy is changed on this evidence alone. Before choosing
a general replacement, extend beyond the partial-predicate and four-register
TopK tests below to pressure/spills, calls, and other representative affected LLKs.
Other chips, arbitrary raw opcodes, formal equivalence, exhaustive input coverage,
and corpus-wide performance are not established here. The proposed unknown-value builtin
remains a design alternative, not an experimentally evaluated implementation.

## Strengthened experiment

The current fixture replaces uniform outputs with BF16 values `1 + Row/128`
and checks SFPSTORE's exact packed address mapping. Typed temporary load/store
addresses match each row. All device cases use the harness's existing sentinel
clearing and at least two executions. This prevents a stale uniform output from
concealing an unwritten row.

Coverage now includes:

- Each L0–L7 separately, with typed work after and also live before the raw write.
- Mixed active/inactive lanes: negative source lanes receive the row value;
  positive source lanes retain 3.0. Every element is checked in position.
- Threaded-value relocation while partially predicated. The inspected ELF
  really contains save/restore SFPMOV instructions with all-lanes modifier 2;
  this is not merely a source-level request for a move.
- Dead raw outputs: only the preexisting typed value is observed. Scheme 3 in
  this test means a discarded read, not a value with a later consumer.

First strengthened O2 run: **230 PASS, 10 XFAIL**, no unexpected failures,
240 cases in 106.99 seconds. XFAIL denotes observed diagnostic corruption, not
correctness success. All effect cases and all genuinely threaded cases passed.
These are individual-register tests, not eight simultaneously live registers.

The compile-only dead-output matrix completed **384/384 compilations** across
registers, TT/TTI, O2/O3, scheduling, and selected pass-disabled controls. Identity
pairs protect the typed value in this point-clobber test, even with the new pass
disabled. A discarded read survives the dump named `optimized`, then the target's
late GIMPLE `rvtt_dce` deletes it before RTL expansion. The pass dump explicitly
reports `Deleting unreachable __builtin_rvtt_sfpreadlreg (0);`. Its assembly
matches the unannotated control in all 128 comparisons.
The unannotated allocation happened to use L0, so L1–L7 non-collisions are not
proof of protection.

The separate compile-only `raw_lreg_threading_pressure.cpp` tests simultaneously
live values: 2, 4, or 7 raw values plus a typed temporary compile for effects and
threading (the latter with the pass disabled), TT/TTI and both scheduler settings:
24 successful compilations. Eight live raw values plus a temporary produce the
expected register-capacity diagnostic in all eight configurations. Those are
rejections, not runtime correctness passes. There is no hardware pressure result.

The scheduling review found no reproduced marker-association bug. GCC treats
volatile UNSPECs and volatile assembly as register barriers; the inspected
scheduled MMIO case keeps the raw store adjacent to its effect marker. The
pass's preceding-instruction lookup remains an assumption deserving broader
coverage, not an established wrong-code finding.

There are distinct obligations:

| Obligation | What current evidence supports |
|---|---|
| Preserve raw output until a later raw consumer | A live threaded value or effect interval; independent pairs fail the L0 gap |
| Protect a typed value from a dead raw output | Identity pair or effect clobber; discarded read is insufficient |
| Carry state across independent wrappers/helpers | Requires a shared state interface or compiler support; not solved by local identity pairs |

The reproducible full device matrix is now:

```sh
bash ../corpus/run_raw_lreg_device.sh /absolute/path/to/new-results
```

Run from `python_tests`, or invoke the script by absolute path from anywhere.
Set `TT_LLK_EXTRA_COMPILER_OPTIONS` to backend/header overrides if necessary.
The runner records separate logs, JUnit results and build directories for O2/O3,
default/explicit scheduling, and pass enabled/disabled. Disabled-pass groups
exclude effect cases because those markers require their implementing pass.
It runs sequentially, does not reset a board, and returns failure if any group
fails. Use a fresh results directory; an existing path reuses/overwrites its
named outputs. Base compiler options should select backend/headers, not override
scheduling. The runner now explicitly enables or disables the live-in pass and
records compiler options in each log.

The original strengthened suite (schemes 0–3, test commit `3a670507bee`)
completed the full matrix. Parsed JUnit records confirm:

| Optimization/scheduling | Pass enabled: PASS / XFAIL | Pass disabled: PASS / XFAIL |
|---|---:|---:|
| O2 / default | 230 / 10 | 166 / 10 |
| O2 / explicit scheduling | 230 / 10 | 166 / 10 |
| O3 / default | 230 / 10 | 166 / 10 |
| O3 / explicit scheduling | 230 / 10 | 166 / 10 |

Total: **1,584 PASS, 80 diagnostic XFAIL, zero failures or errors**, 1,664 cases.
All 80 JUnit skipped entries have type `pytest.xfail`; they are observed wrong
outputs in diagnostic controls, not unexecuted hardware cases. Each device case
uses at least two executions with sentinel clearing. This is not an LLK corpus
or performance result. Later-added schemes have separate measurements below.

Current evidence logs are on quietbox under `/tmp/lreg-review.Y7HFRq`:
`strengthened.log`, `matrix.log`, `matrix/`, and `dead-*` compiler artifacts.
These temporary paths are evidence locations, not required reproduction paths.

Production integration remains open: TopK carries coupled value/index banks;
Welford carries state across helpers and loops; LOADMACRO has configured effects
not recoverable from its word alone. A wrapper-local rewrite is not justified.

## Combined annotations and the old-destination counterexample

Scheme 4 retains the producer capture with `saved = readlreg(R)` followed by
`writelreg(saved,R)`, then uses the same `saved` at the consumer. Unlike a
discarded read, the initial pair survives when the output is dead. This combines
point-clobber modeling with a genuine output lifetime; it is still a caller-owned
region interface, not a substitute inside independent macros.

The new `test_raw_lreg_old_destination` inserts typed work between initializing
the raw destination to 3.0 and partially overwriting it. Output-only threading
(schemes 3 and 4) does not model this earlier implicit input. On L0, both TT and
TTI lose **512/1024 elements**, exactly the inactive half. Effect annotations
preserve that input lifetime. Scheme 5 explicitly captures the old destination,
threads it across the early gap, and restores it before the partial raw write;
it then uses the retained output-threading pattern. Both mechanisms pass this
counterexample with the implementing pass enabled; the explicit-state version
also passes with it disabled.

The follow-up selection is `-k 'pinned or old_destination'`; when the pass is
disabled, use `-k '(pinned or old_destination) and not effects'`. This selects
the new cases without presenting a repeated original matrix as additional
coverage. The full runner also includes them automatically. O2/O3 with explicit
scheduling, pass-enabled/disabled follow-up results in `combined-*.log` and
`combined-*.xml`:

| Follow-up configuration | PASS | Diagnostic XFAIL |
|---|---:|---:|
| O2 scheduled, pass enabled | 140 | 4 |
| O2 scheduled, pass disabled | 124 | 4 |
| O3 scheduled, pass enabled | 140 | 4 |
| O3 scheduled, pass disabled | 124 | 4 |

Total **528 PASS, 16 diagnostic XFAIL**, zero failures/errors, 544 cases.
All 64 scheme-5 cases have zero input/output mismatches. This follow-up does not
establish default-scheduling coverage for every new scheme; do not combine it
with the older matrix as though the schemes had identical configuration coverage.

A fresh full-current-suite O2/default-scheduling/pass-enabled run finished
**370 PASS, 14 diagnostic XFAIL**, 384 cases in 173.56 seconds (`final-suite.xml`,
`final-suite.log`). This establishes that configuration for all current cases.
A separate eight-case L0 observation run (`lane-split.log`) logs the mask split:
post-only variants have **zero active-lane errors and 512 inactive-lane errors**;
effect and full input/output-threaded variants have zero in both classes. This
is direct measurement of which lanes fail, not an inference from the total.

The resulting annotation obligations are:

1. Before a raw operation, preserve and supply every input, including old
   destinations whose inactive lanes survive.
2. After a raw write, retain a point definition/clobber even if its output is
   otherwise unused.
3. Carry each live result through C++ state to subsequent consumers, updating
   that state after every raw write rather than restoring stale entry values.

The existing effect interface encodes these obligations through masks and the
compiler pass. Explicit state can encode them at region boundaries instead.
The tests support neither removing the pass globally nor claiming it is the
only possible solution. No production policy changes or new compiler passes
were made in this experiment.

## Production-linked TopK merge experiment

`sources/topk_threaded_merge.h` adapts only the production merge region:
four raw loads, coupled value/index swap, four raw stores. It captures and
threads L0/L1/L4/L5 through C++ values, recapturing all four after the swap.
The outer loop, formats and address arithmetic follow the production header;
local sort and rebuild are unchanged. Raw operations in the adapted region
bypass effect wrappers. No public LLK header or compiler implementation changes.

`TOPK_IMPL=2` is the measured adaptation. `TOPK_IMPL=3` additionally inserts a
typed load/store roundtrip of the OTHER compare operand after capturing L0.
This keeps the input unchanged while making accidental L0 reuse observable;
it is correctness stress only, never a performance arm. An earlier version
reloaded the same operand, which could hide corruption; the recorded final
stress runs below use the distinct operand.

Same Blackhole/runtime/compiler identities as above, live-in pass disabled:

| Configuration | Exact correctness |
|---|---:|
| O2 default scheduling, hand + threaded | 24 PASS |
| O3 explicit scheduling, hand + threaded | 24 PASS |
| O3 explicit scheduling, hand + threaded + distinct-value stress | 36 PASS |
| Same, diagnostic complete-unroll limit300 | 36 PASS |

The 36-case matrix covers two directions, three orders (ascending, descending,
seeded permutation), 32/64 rows, width128, K32, unique finite BF16 values,
and non-stable sorting. Both value and index tiles must equal the independent
golden tensor exactly. Each case executes at least twice with sentinel clearing.
This does not cover ties, special values, other formats/chips, or all TopK phases
rewritten using threaded state.

Five-run TOPK_BODY profiling measures the whole 32x128 pipeline, not isolated
merge latency. Same O3/scheduling/pass-disabled options for both arms:

| Complete-unroll size limit | Hand cycles | Threaded cycles | Threaded vs hand |
|---|---:|---:|---:|
| Default (200) | 5038 | 5186 | +2.94% |
| Diagnostic control (300) | 4941 | 4827 | -2.31% |

The default threaded GIMPLE dump explicitly refuses full unroll: estimated
size280 minus27 eliminated =253, above200. Its sixteen read/write annotations
each cost one GIMPLE unit. The hand ELF has eight unrolled swap sites; the
threaded ELF retains one swap plus scalar address calculations and a backedge.
Neither default ELF contains SFPMOV. This establishes a code-generation
difference, not attribution of every cycle to one loop.

The 300 limit is a diagnostic compiler-option intervention, NOT a shipped
default or a production win claim: it also changes other loops, and the hand
timing changes too. Both arms must always receive the same setting. The
control ELF confirms eight straight-line swaps, constant offsets, no merge
backedge, and zero SFPMOV instructions. The existing
full-stack launch-flatten implementation prices these markers at zero but
requires positive-cost typed content, so it does not automatically cover this
raw-delivery-plus-annotations case. It was not the compiler used for these runs.
Any targeted cost-model/eligibility change requires its own compiler regression
tests and runtime validation; no new pass is introduced here.

### Reproduce TopK

From `tt_metal/tt-llk/tests/python_tests`, with the matching backend and headers
selected as in the earlier re-run section:

```sh
export CHIP_ARCH=blackhole
# Prepend -B/path/to/backend/ -I/path/to/matching/sfpi/include if needed.
export TT_LLK_EXTRA_COMPILER_OPTIONS='-O3 -fschedule-insns -fschedule-insns2 -fdisable-rtl-rvtt_lreg_livein'
../.venv/bin/python -m pytest test_topk.py::test_topk_threaded_merge_exact -x -s -q
../.venv/bin/python -m pytest test_topk.py::test_topk_device_profile \
  -k 'handwritten or threaded_merge' -x -s -q

# Diagnostic control: same source and compiler; apply to BOTH arms.
export TT_LLK_EXTRA_COMPILER_OPTIONS="$TT_LLK_EXTRA_COMPILER_OPTIONS --param=max-completely-peeled-insns=300"
../.venv/bin/python -m pytest test_topk.py::test_topk_threaded_merge_exact -x -s -q
../.venv/bin/python -m pytest test_topk.py::test_topk_device_profile \
  -k 'handwritten or threaded_merge' -x -s -q
```

Profiling checks valid counters, not correctness or a speedup threshold; run
the exact gate first. Existing quietbox evidence under `/tmp/lreg-review.Y7HFRq`:
`topk-exact`, `topk-scheduled`, `topk-profile`, `topk-unroll-control`,
`topk-distinct-stress`, `topk-distinct-control` (logs/XML as applicable). Temporary evidence paths are
not reproduction prerequisites. Add `-fdump-tree-cunroll-details` to inspect
the unroll decision; use a separate RUNNER_TEMP per run to retain its ELFs.

## Public LRegFile integration — 2026-10-08

The test-only production merge adaptation now uses public `sfpi::l_reg`
reads and assignments instead of direct read/write builtins. Value registers
use `vFloat`, index registers use `vUInt`; neither is numerically converted.
Every coupled result is recaptured after the raw swap. The stress-only typed
load/store remains a builtin call to preserve its exact address-mode spelling.
No new public API, compiler pass, production-default selector, or unroll
pragma was introduced.

Fresh tests use the same standalone `bc27e710ef4` compiler with the live-in
pass disabled. The two-format matrix adds IEEE FP16 to BF16: handwritten,
public-API threaded merge, and distinct-value stress; two row shapes, two
directions, three deterministic orders, K32/width128/non-stable. O2/default
scheduling and O3/explicit scheduling each completed **72 PASS**, zero
skips/failures. These are
exact value AND index checks with at least two executions per case. Stable
sorting/ties, special values, FP32, other widths/K values, other chips, and
other TopK phases rewritten with this API remain outside the claim.

Before expanding to two formats, the 36-case BF16 correctness matrix and two
five-run profiling arms passed with each of the default and diagnostic unroll
settings. Fresh public-API timings reproduce the earlier builtin measurements:

| Complete-unroll limit | Hand cycles | Public-API threaded cycles |
|---|---:|---:|
| Default (200) | 5038 | 5186 |
| Diagnostic (300), same flags on both arms | 4941 | 4827 |

The public interface therefore works in this tested region, but does not solve
the default **2.94% regression**. Do not enable the adaptation as a production
default or use the diagnostic global knob as a claimed targeted compiler fix.
The local, unbuilt launch-flatten eligibility patch was NOT used for any run.

The reproduction commands above now select both formats automatically.
Fresh evidence at the same quietbox root: `topk-public-api.{log,xml}`,
`topk-public-api-control.{log,xml}`, `topk-public-formats.{log,xml}`, and
`topk-public-formats-O2.{log,xml}`.
These are separate from the upstream-based macro companion PR: passing the
explicit-state region does not validate every opcode in the effect decoder.

## Targeted compiler fix — 2026-10-08

Compiler commit `064ef4565ea` extends the existing opt-in launch-flatten pass
to raw-delivery loops containing both explicit LREG reads and writes. The
generic size estimate charges these markers even when allocation emits no
moves. Pure raw loops and replay-owner-only loops remain excluded; proven
trip counts, body admission, word/function budgets and QSR exclusion are
unchanged. No new pass, global size-limit change, or kernel-name special case.

Fresh Linux builds compare baseline `846eae4da23` against that one patch.
On Blackhole, with the public-API TopK source, O3, explicit scheduling,
live-in pass disabled, and the same launch-flatten flag for both arms:

| Compiler / launch-flatten | Hand cycles | Public-API threaded cycles | Exact cases |
|---|---:|---:|---:|
| Baseline / on | 5038 | 5186 | 72 PASS |
| Patched / off | 5038 | 5186 | 72 PASS |
| Patched / on | 5038 | 4929 | 72 PASS |

Five profiling executions per arm; whole TOPK_BODY, not isolated merge timing.
The targeted change removes the observed regression: threaded is **2.16%
faster than hand** in this workload. Its dump requests complete unrolling of
loop39, nine delivery words per trip, eight trips. The global diagnostic
`max-completely-peeled-insns=300` option is NOT present. Patched O2/default
scheduling with launch-flatten enabled also passes all 72 exact cases.

Compiler tests: new positive regression fails its two unroll assertions before
the patch (50 PASS / 2 FAIL in the focused suite), then the complete focused
suite passes (52 PASS / 0 FAIL). Fresh full `rvtt.exp` with SFPI enabled:
**8,085 PASS, 2 expected XFAIL, zero unexpected failures**, including 1,537
SFPI checks. Expected failures are the two `rv/raw-race-wh.C` assembly scans.

To reproduce the fixed configuration, use the re-run commands above with the
patched compiler/backend and append **`-mtt-tensix-optimize-launch-flatten`**
to TT_LLK_EXTRA_COMPILER_OPTIONS (do not append the global diagnostic limit).
For the control, replace that option with
`-mno-tt-tensix-optimize-launch-flatten`. Run both implementation arms with
the same options. Compiler checks, from the GCC build's `gcc/` directory:

```sh
SFPI=/path/to/matching/installed-sfpi make -k check-g++ \
  RUNTESTFLAGS='rvtt.exp=launch-flatten*.C'
SFPI=/path/to/matching/installed-sfpi make -k check-g++ RUNTESTFLAGS='rvtt.exp'
```

Inspect `g++.sum`: make's exit status alone did not report the baseline scan
failures. Evidence under the same quietbox root: `flatten-baseline-tests/`,
`flatten-patched-tests/`, `flatten-full-tests/`, `topk-flatten-baseline`,
`topk-flatten-patched`, `topk-flatten-off`, and `topk-flatten-patched-O2`
(logs/XML/build artifacts as applicable). Compiler build logs are
`fullstack-baseline-build.log` and `fullstack-patched-build.log`.

This resolves the measured performance blocker with an existing compiler knob.
It does not expand the input/architecture coverage stated above, enable the
test adaptation as a production default, validate all raw opcodes, or justify
removing the live-in pass globally.

## Production interface decision — 2026-10-08

For the validated Blackhole merge region, use caller-owned values through
the existing public `sfpi::l_reg` API. No new SFPI builtin or public class
member is needed. The production header now exposes
`_bitonic_topk_merge_explicit<APPROX, DEST_ACC, TOP_MIN, STABLE>(m_iter, k)`.
It shares the original production loop/addressing implementation and selects
`bitonic_topk_merge8_explicit` for the four-register load/swap/store region.
The existing `_bitonic_topk_merge` entry and its default behavior are unchanged.
Only Blackhole is changed; this is an opt-in interface, not an automatic
production rollout.

All L0/L1 values and L4/L5 indices are captured immediately after their loads,
restored for the swap, then recaptured as new coupled results before stores.
The region issues raw words directly, so its correctness does not borrow
protection from the macro-effect decoder. `TOPK_IMPL=2` now invokes this actual
production entry rather than a separate test-only implementation. The
injected-temporary variant remains a test-only stress copy, not the timed path.

Fresh production-entry results with compiler `064ef4565ea` and launch-flatten
enabled: 72 exact BF16/FP16 cases pass at O3/explicit scheduling with the
live-in pass disabled; another 72 pass with that pass enabled. Both profiling
runs give legacy 5038/explicit 4929 cycles, five executions per arm. Both final
runs use the named entry above. The O2 run also passes all 72 cases with the
live-in pass disabled.

Adoption contract: retain the surrounding TopK initialization/lane-state and
address-mode contract, and enable `-mtt-tensix-optimize-launch-flatten` using
the fixed compiler to avoid the known cost regression. Validated input scope
remains K32, width128, rows32/64, finite unique BF16/FP16 values, non-stable,
both sort directions. No tie/special-value/FP32/general-predicate claim is
added by placing the interface in a production header.

Other raw users are NOT mechanically rewritten. Helpers spanning separately
owned register state need an explicit state-passing interface; LOADMACRO needs
its configured effects, and predicated writes need old inactive destinations.
The global macro-effect proposal remains draft; this region is not evidence
that either mechanism is universally necessary or sufficient.

Reproduce with the existing test commands and compiler option above. Final
evidence: `topk-production-final.{log,xml}`, `topk-production-enabled.{log,xml}`,
and `topk-production-O2.{log,xml}` under the same quietbox evidence root.

## All-region rollout: EMA and Welford — 2026-10-08

This rollout is incomplete. `inventory_raw_sfpu.py` inventories source spellings
in the three architectures' `common/inc/sfpu` trees, including experimental
headers and inactive preprocessor branches. With comments and ordinary string
literals excluded it finds 39 headers (18 Blackhole, 11 Wormhole, 10 Quasar)
and 2004 source sites. The preliminary grep count of 44 included comment-only
matches. These are not instantiated-kernel or validated-region counts, and the
inventory does not close calls into helpers outside those trees.

```sh
python3 corpus/inventory_raw_sfpu.py --self-test
python3 corpus/inventory_raw_sfpu.py
```

EMA now has an opt-in shared production header,
`common/ckernel_sfpu_ema_explicit.h`. The existing corpus entry points forward
to it rather than keeping a second implementation. Caller-owned `EmaState`
carries values across tile calls; existing production defaults are unchanged.
Contract 1 is the FMA ordering tested here. Contract 2 retains an existing
different arithmetic ordering and is NOT admitted as a bit-exact replacement.

On Blackhole with compiler `064ef4565ea`, O3, explicit scheduling,
launch-flatten enabled and the raw-LREG live-in pass disabled, the production
EMA module passes 18 tests. Exact hand/typed comparisons cover 1, 2, 4 and 32
tiles, seed 0, BF16 finite input in [-4,4], alpha=0.25 and beta=0.75; both arms
also pass the independent recurrence oracle. This does not cover arbitrary
coefficients, special values, FP32 or other architectures.

Welford's existing typed candidate passes 26 tests under the same compiler
options, including six new hand-replay/typed-direct exact comparisons at
prefix lengths 1, 2, 4, 8, 16 and 32. Exactness here is of the captured BF16
mean/M2 outputs, NOT all internal FP32 bits or scratch registers. Both arms
also satisfy the existing independent mean/M2 tolerance checks. Input seed is
20260814, finite BF16 in [-4,4]. No new production Welford default is adopted.

Matched device MATH-zone measurements, five executions per arm:

| Region | Hand | Typed | Interpretation |
| --- | ---: | ---: | --- |
| EMA, one tile | 335 | 329 | 1.79% faster |
| EMA, 32 tiles, cycles/tile | 212.21875 | 209.125 | 1.46% faster |
| Welford, 32 rows | 325 | 350 | 7.69% slower than hand replay |

EMA initialization is outside the timed body. Welford hand-direct is 464.8
cycles, but the faster replay implementation is the relevant baseline; using
hand-direct would misleadingly report a win. Welford needs replay/codegen
work before a non-regressing replacement. No blanket raw-macro rewrite is
justified by these results, especially inside fixed-length recorded streams
or configured LOADMACRO regions.

The EMA profiler previously mixed parameter schemas and selected earlier
cases' rows when run as a module. All EMA arms now carry `EMA_IMPL`, and row
selection includes implementation and tile count. Welford profiling likewise
selects its implementation. Both modules support a single combined invocation
and retain fractional five-run means. The initial EMA profiling attempt failed
in the harness; the table comes from the subsequent successful production run.

From `tests/python_tests`, with `CHIP_ARCH=blackhole`, matching compiler/header
paths and these extra options:

```sh
export TT_LLK_EXTRA_COMPILER_OPTIONS="-B$GCC_BUILD/gcc/ -I$SFPI/include -O3 -fschedule-insns -fschedule-insns2 -fdisable-rtl-rvtt_lreg_livein -mtt-tensix-optimize-launch-flatten"
../.venv/bin/python -m pytest -s -q test_sfpu_ema.py
../.venv/bin/python -m pytest -s -q test_sfpu_welford_prefix_snapshot.py
```

Set `GCC_BUILD` to the built compiler directory and `SFPI` to matching installed
SFPI before exporting the options. On the measured node those are
`/home/ttuser/craq-build/sfpi/build/build-gcc-newlib-stage1` and
`/tmp/sfpi-lreg-test.Mg1BuW/build/sfpi`. Evidence in the quietbox root above:
`ema-production-state.{log,xml}` and `welford-state-all.{log,xml}`, with build
and profiler CSV artifacts in the corresponding run directories. The Parquet
writer warns that two Welford columns are dropped from its narrower schema;
the full per-implementation rows are retained in the CSV/log evidence.

Remaining scope includes other TopK regions, configured LOADMACRO users,
recorded streams, experimental kernels and non-Blackhole silicon. These
results neither finish the all-region migration nor prove universal safety.
