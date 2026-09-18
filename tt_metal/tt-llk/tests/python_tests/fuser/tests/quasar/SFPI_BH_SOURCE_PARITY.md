# BH-source-copy cleanup and validated exceptions

User requirement: Quasar SFPI kernels should be carbon copies of Blackhole
where possible. Preserve source formulas, coefficients, control flow, comments,
and API shape; keep only justified architecture/compiler adaptations.

Active worktree: `/localdev/vvukomanovic/tt-metal-sfpi-fuser-sweep`.
Base: `marko/fuser-sweep`, `1b1ac4875571ceb230ba62705d32a252ddb3e657`.
The 168 YAML files, 278 variants, and 762 stages are unchanged. Fuser sweep
machinery, goldens, tolerances, and simulator infrastructure were not changed
for this cleanup.

## Validation — passed

The final source passed all **278 parity variants / 762 stages**, plus three
existing fuser controls, on the Quasar simulator on 2026-09-17. The exact-ID
audit found no missing, duplicate, or nonpassing cases. Host verification also
confirmed all 168 YAML files and 762 original input tensors were preserved.
See [coverage and batch evidence](SFPI_BH_COVERAGE.md).

The repository debug-kernel and run-test skills governed diagnosis and testing:
unchanged reproduction before fixing numerical failures, serialized bounded
emulator runs, and no automatic retry/reset for the environment failure.

### Diagnostic history

On 2026-09-17, all 278 variants compiled successfully after the changes.
The focused simulator run passed eight of 14 variants:

- Passed: MAC, I1, and Tanhshrink in both formats; BinaryPow FP32; Rpow FP32.
- Failed: all four SigmoidAppx variants and Xielu BF16/FP32 (numerical mismatches).
- The first unchanged six-case reproduction attempt executed zero cases: emulator
  startup failed with `WRP0625E`, resources on `soc-zebu-03` occupied by another
  user (`mvlahovic`, session `c5acd5bc`). No retry or reset was performed.
- After the user freed the emulator, all six numerical failures reproduced.
- Minimal SigmoidAppx initialization and Xielu store-placement fixes then passed
  all six cases. The final 278-variant suite and three controls subsequently passed.

These new full-suite results supersede the previous revision's green run.
No commits were made. Only comments and documentation changed after testing.

## Restored source and verified exceptions

- MAC executable source now matches BH exactly: recorded/replayed MAD rather
  than a replacement loop. Quasar disassembly confirms five FP32 / six BF16
  recorded instructions, without swallowing subsequent synchronization. Both
  runtime variants pass.
- I0's original polynomial macro/body and I1's original main control flow are
  restored. I1 BF16/FP32 passed the focused runtime run.
- Xielu retains BH's branch order, formulas, coefficients, and BF16 rounding.
  Its MAD helper returns a value and the Dest store occurs once after the outer
  predicate chain: original branch-local stores reproducibly failed; this
  minimal change passes both formats.
- Erfc's original table, comments, and calculation body are restored. Erf and
  Digamma retain original tables/body with template-selected LUT traits because
  fuser selects precision per operation, not through `INP_FLOAT32`.
- Fixed tile offsets of 32 and original iteration defaults are restored where
  identical to Quasar's constants. Unnecessary formatting/comment rewrites were
  removed from the other headers.
- Long polynomial calls explicitly use `eval<100>` to select BH's serial Horner
  order. Quasar's existing shared evaluator defaults to an even/odd split at
  six coefficients. Its global default was not changed.
- FP32 BinaryPow, Rpow, and Tanhshrink retain register-lifetime adaptations:
  their original bodies failed Quasar compilation with
  `too few lregs to hold live values`. Logs identify the failed originals;
  corrected versions compile and pass the focused runtime cases.

Other remaining architecture/interface differences include SM32 integer Dest
access, unsupported narrow-cast substitutions, include/init/helper spelling,
missing UInt32/Bfp8_b enum alternatives, precision-template selection, Exp2
range-reduction/predicate adaptation, local copies of required BH log/tanh/exp
helpers, SFPI trunc/floor helpers, and the non-advancing integer-to-float division
body used by its test adapter. These are not claims of bitwise BH equivalence.

### Remaining substantive exceptions

| Kernel | Difference retained |
| --- | --- |
| Exp2 | Target rounding/conversion and exponent-range predicates; original minimax coefficients |
| BinaryPow / Rpow FP32 | Magnitude/finalization split and input reloads to avoid demonstrated LReg allocation failures |
| Tanhshrink | Branch ordering, lifetimes, reloads, and separate stores to fit available registers |
| Xielu | One final Dest store; a demonstrated compiler/code-shape workaround, not an ISA prohibition on predicated stores |
| SigmoidAppx | Quasar LUT API and configuration-register loading; original coefficients |
| DivInt32-to-float | Non-advancing arithmetic body extracted for the test adapter |

Dependency parity matters too: I1, Tanhshrink, BinaryPow/Rpow, and SiTU-GLU use
the existing Quasar exponential helper, whose rounding/control flow is not a
literal BH copy. Local BH-derived log/tanh/exp helpers, target trunc/floor helpers,
and the supported subset of piecewise-rational evaluation are also adaptations.
The shared Quasar exponential implementation was not replaced for this task.

Literal file equality is narrower than calculation-body equality: of the 63
changed metal SFPU headers, 61 have same-path BH counterparts, and only Identity
is byte-identical. Many others differ only in required includes, initialization,
representation, precision selection, or helper names. MAC executable source is
identical despite two architecture-comment differences. This work does not
claim that all headers or all generated instructions are 100% identical.

Erf now preserves BH's first two template arguments, `<APPROX, ITERATIONS>`,
and appends the explicit precision argument. Its default honors `INP_FLOAT32`.
The test dispatcher supplies the appended argument, as it does for Digamma.

## Minimal runtime fixes

### SigmoidAppx

The original four-argument BH `sfpi::lut` API does not compile for Quasar.
The loop now uses `sfpi::lut<sfpi::LutMode::Fp8x3>(val) + 0.5f`, with the same
BH coefficient constants programmed into Quasar LUT configuration registers.

The initial runtime failed: the output behaved like a step function. Disassembly
showed the first two coefficient writes folded into immediate-mode `SFPCONFIG`
to CREG9/10, while the third was loaded through an LReg. The existing Quasar GELU kernel uses
an explicit configuration-write workaround for those constant registers.

Applied change: use `math::_sfpu_load_config32_` for all three
`sfpi::sLut8si(...).get()` coefficient words, retaining the BH coefficients and
LUT loop. All four unchanged tests pass. Goldens and tolerances are unchanged.

Fresh b03 BF16 and FP32 disassembly verifies coefficient words `0x00003dff`,
`0x000021d8`, and `0x0000ff10` loaded through LREG0 followed by
`sfpconfig 9,0,0`, `sfpconfig 10,0,0`, and `sfpconfig 11,0,0`. The calculation
still emits SFPLUT (modifier 4) followed by addition of 0.5.

### Xielu

With original branch-local stores, FP32 negative cases can retain their inputs;
BF16 also exhibits lane-pattern corruption. This suggests predicate/store
lowering around the nested exponential helper, not changed coefficients.
That diagnosis is not yet proven.

The smallest experiment succeeded: return the result from `_xielu_mad_`, assign
it in the original branches, and store once after the outer `v_endif`. Both
formats pass. No exponential hoisting, scalar-alpha conversion, input reload,
or larger lifetime rewrite was needed.

## Local evidence

Artifacts: `/tmp/llk-sfpi-carbon-copy.atUjG9/`.

| Invocation | Result |
| --- | --- |
| compile01 | BH four-argument LUT API rejected |
| compile02 | 57 compiled; BinaryPow FP32 LReg allocation failure |
| compile03 | 261 compiled; Rpow/Tanhshrink FP32 LReg failures |
| compile04 | All 278 compiled, 38.31 seconds |
| focused01 | 8 passed, 6 failed, 100.83 seconds |
| repro01 | Environment startup blocked; zero cases executed |
| repro02 | All six numerical failures reproduced, 82.29 seconds |
| focused02 | All six fixed variants passed, 59.74 seconds |
| b01–b08 | All 278 parity variants / 762 stages passed, 650.62 seconds total |
| b09 | All three existing fuser controls passed, 49.43 seconds |

Compile-only wrapper currently hardcodes `-x`; `--maxfail 0` does not override
that phase. This explains early compile stops; no wrapper changes were made.

`before.tar.gz` preserves the pre-cleanup files. Original sweep batch selectors
are in `/tmp/llk-sfpi-sweeps.rWeZkb/batches.json`. All eight bounded parity batches
and the three existing controls ran through the repository run-test skill. The
runtime audit used these new logs, not the historical consolidation logs:

```bash
python3 /tmp/llk-sfpi-sweeps.rWeZkb/audit_runtime.py --require-complete \
  --log-pattern '/tmp/llk-sfpi-carbon-copy.atUjG9/b{number:02d}/run.log'
```
