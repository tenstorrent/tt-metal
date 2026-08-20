# Measurements — `sqrt_custom(+inf)`, accuracy and performance

Accuracy and performance of the two kernels this branch changes, measured against the branch
point rather than recalled from the original development run. Read-only record: no code, no
test changes.

| | |
|---|---|
| Branch | `ldjurovic/sfpu_52930_sqrt_custom_infinity` |
| Baseline | `origin/main` @ `35ec0aba7a8` — the branch point, so the only kernel delta is this branch's |
| Silicon | Wormhole n300 (UMD chip 0, board `010001461…`) |
| Blackhole | compile-verified only — no BH hardware on this host |
| Date | 2026-08-20 |

Companion to [RESULTS_52930_sqrt_custom_infinity.md](RESULTS_52930_sqrt_custom_infinity.md),
which records the same fix as it was developed. This one re-measures it against the branch
point after the split, so the figures are attributable to this branch alone. Where the two
overlap they agree — `SqrtCustom` +9.69 % there against +9.76 % here, `Erfinv` +4.10 % against
+4.11 % — the small drift being a different `main` underneath, not a different kernel. Section
7 is where they genuinely differ, and says why.

Kernels changed, and nothing else:

- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt_custom.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt_custom.h`

`sfpu_sqrt_custom` is a shared helper, so the ops under measurement are the direct op
(`SqrtCustom`) and its three consumers: `Erfinv`, `Asin`, `Acos`. `Acosh` and `Asinh` are
carried through every table as controls — same suite, same shape, no `sqrt_custom` in them.

---

## 1. Method

Both trees are checkouts of the same repo with isolated build roots (`RUNNER_TEMP`), so
neither can serve the other a stale ELF. Every measurement below is baseline and branch run
back to back on the same board, same session, same toolchain.

**Accuracy** — raw hardware bit patterns, not tolerance comparisons. Two probes drive a fixed
stimulus list through the kernel and dump the `res` buffer as hex, so a before/after pair
diffs byte for byte. The stimulus is a literal list in the probe rather than `edge_values()`,
so both trees are handed byte-identical input regardless of what their `SPECIALS_READY_OPS` /
`specials_safe` tables say — the branch changes those tables, and a probe that read them would
be comparing two different experiments.

- `test_sfpu_wh_sqrtcustom_numerics.py` — 21 values spanning the exponent range plus the
  poles, on all 8 (format pair, dest_acc) combinations; `Asin`/`Acos` on 16 in-domain values.
- `test_sfpu_wh_erfinv_numerics.py` — 18 values, the ±1 poles plus in-domain, all 8
  combinations. Written for this report: the existing probe never drove `Erfinv`, which is the
  consumer the defect was found through.

Both are instruments, not merge candidates, and are not committed on this branch.

**Performance** — `perf_eltwise_unary_sfpu.py`, the CI harness with CI's flags
(`--speed-of-light`, producer then consumer). Figures are `MATH_ISOLATE` on the `TILE_LOOP`
marker divided by `tile_cnt × loop_factor` (8 × 16 = 128), i.e. cycles per tile. Three
consumer runs per tree over 24 variants each.

**Instruction accounting** — `riscv-tt-elf-objdump -d` on the `MATH_ISOLATE` `math.elf` of each
variant, paired by build hash across the two trees.

---

## 2. Accuracy — `SqrtCustom`

Every finite input is bit-identical, baseline vs branch, on all 8 combinations. That is the
whole justification for this shape of fix, and it is checked against raw bit patterns rather
than against a tolerance. The complete diff of the two records is four rows, all of them
non-finite:

| Input | Baseline | Branch | IEEE | Where visible |
|---|---|---|---|---|
| `+inf` | `0x7FC00001` (NaN) | `0x7F800000` (`+inf`) | `+inf` | the 2 combinations that carry an fp32 datum end to end |
| `-inf` | `0x7F800000` (`+inf`) | `0xFF800000` (`-inf`) | NaN | all 8 |
| `NaN` | `0x7FC00001` | `0x7FC00000` | NaN | `Float32→Float32` `dest_acc=Yes` |
| `-0.0` | `0x00000000` | `0x00000000` | `-0.0` | unchanged, pre-existing |

`+inf` reads as repaired on only 2 of 8 combinations because on the other 6 the packer already
narrowed the NaN to `+inf` on the way out (`SFPSTORE`: "NaN is also converted to infinity"),
so those agreed with the golden by accident before and agree for the right reason now.

`-inf` is the deliberate limit of the minimal fix, and the direction matters: the guard passes
non-finite input straight through, which is right for `+inf` and NaN and wrong for `-inf`. The
branch does not hide this — it enrols `SqrtCustom` in `SPECIALS_READY_OPS` and records the
divergence as a per-combination xfail with its reason. Note that this makes a real
disagreement *visible* rather than introducing one: before the fix that combination returned
`+inf`, which agreed with the golden only because the golden's NaN is itself narrowed to `inf`
on a bf16 output.

`Float32→Float32`, `dest_acc=Yes` — the pipeline where nothing is narrowed — in full:

```
                              baseline        branch
x=0x3F800000 (1.0)          0x3F7FFFAC      0x3F7FFFAC     rel 5.007e-06
x=0x40800000 (4.0)          0x3FFFFFAC      0x3FFFFFAC     rel 5.007e-06
x=0x40000000 (2.0)          0x3FB504F3      0x3FB504F3     rel 1.711e-08
x=0x3E800000 (0.25)         0x3EFFFFAC      0x3EFFFFAC     rel 5.007e-06
x=0x40400000 (3.0)          0x3FDDB3CC      0x3FDDB3CC     rel 7.750e-07
x=0x41100000 (9.0)          0x403FFFEE      0x403FFFEE     rel 1.431e-06
x=0x0DA24260 (1e-30)        0x26901D7C      0x26901D7C     rel 1.023e-07
x=0x7149F2CA (1e+30)        0x58635FA8      0x58635FA8     rel 8.012e-08
x=0x3FC00000 (1.5)          0x3F9CC459      0x3F9CC459     rel 2.300e-06
x=0x42C80000 (100.0)        0x411FFFE2      0x411FFFE2     rel 2.861e-06
x=0x3A83126F (0.001)        0x3D0186CD      0x3D0186CD     rel 2.528e-06
x=0x2EDBE6FF (1e-10)        0x3727C590      0x3727C590     rel 2.572e-06
x=0x501502F9 (1e+10)        0x47C34FE5      0x47C34FE5     rel 2.109e-06
x=0x00800000 (1.1754944e-38) 0x208AF000     0x208AF000     rel 1.171e+00   <- pre-existing
x=0x7F7FFFFF (3.4028235e+38) 0x600AF000     0x600AF000     rel 1.171e+00   <- pre-existing
x=0x34800000 (2.3841858e-07) 0x39FFFFAC     0x39FFFFAC     rel 5.007e-06
x=0x7F800000 (+inf)          0x7FC00001     0x7F800000     <- repaired
x=0xFF800000 (-inf)          0x7F800000     0xFF800000     <- changed, still not NaN
x=0x7FC00000 (NaN)           0x7FC00001     0x7FC00000     <- canonicalised
```

The two `rel 1.171e+00` rows are the **pre-existing 117 % error at the fp32 extremes** — the
bf16-magic seed runs out of range at the minimum normal and at `FLT_MAX`. Identical before and
after: this branch neither causes nor fixes it, and it wants its own issue.

## 3. Accuracy — `Erfinv`, the consumer the defect was found through

Every finite input bit-identical. The only change is the pole:

| Input | Baseline | Branch | Golden |
|---|---|---|---|
| `+1.0` | `0x7FC00001` (NaN) | `0x7F800000` (`+inf`) | `+inf` |
| `-1.0` | `0xFFC00001` (NaN) | `0xFF800000` (`-inf`) | `-inf` |

Visible on `Float16_b→Float32 dest_acc=Yes` and `Float32→Float32 dest_acc=Yes`. On the other 6
combinations the raw NaN was narrowed to `±inf` by the packer, so they read `±inf` both before
and after — the kernel was wrong on all 8, but only 2 could show it.

Two rows are worth flagging as **unchanged pre-existing error**, so this record is not read as
a clean bill of health for `erfinv`: `erfinv(0.001)` has `rel 1.025e-01` and `erfinv(1e-6)`
returns `0x00000000` against a non-zero golden (`rel 1.000e+00`). Both are identical before
and after. Neither is a sqrt_custom defect and neither is in scope here.

## 4. Accuracy — `Asin` / `Acos`

Bit-identical on every probe value, baseline vs branch. The two records diff to nothing.

For in-domain `|v| ≤ 1` the range reduction hands `sqrt_custom` an argument in `[0, 0.5]`,
which never reaches the guard, so this is the expected result and it is now measured rather
than assumed.

---

## 5. Performance — Wormhole n300, `MATH_ISOLATE` cycles per tile

Three runs per tree. "Separated" means the slowest branch run is still faster than the fastest
baseline run (or vice versa) — i.e. the delta is not inside the run-to-run spread.

| Op | dest_acc | Baseline | Branch | Δ | % | Separated |
|---|---|---|---|---|---|---|
| **SqrtCustom** | No | 983.64 | 1079.65 | +96.01 | **+9.76 %** | yes |
| **SqrtCustom** | Yes | 985.70 | 1081.45 | +95.76 | **+9.71 %** | yes |
| **Erfinv** | No | 3260.04 | 3393.96 | +133.92 | **+4.11 %** | yes |
| **Erfinv** | Yes | 3265.19 | 3395.75 | +130.56 | **+4.00 %** | yes |
| **Asin** | No | 2391.79 | 2487.84 | +96.05 | **+4.02 %** | yes |
| **Asin** | Yes | 1945.49 | 1945.49 | 0.00 | 0.00 % | — |
| **Acos** | No | 2423.84 | 2519.84 | +96.00 | **+3.96 %** | yes |
| **Acos** | Yes | 2042.02 | 2042.02 | 0.00 | 0.00 % | — |
| Acosh *(control)* | No | 3319.95 | 3319.95 | 0.00 | 0.00 % | — |
| Acosh *(control)* | Yes | 3965.50 | 3965.50 | 0.00 | 0.00 % | — |
| Asinh *(control)* | No | 5123.17 | 5123.14 | −0.03 | −0.00 % | — |
| Asinh *(control)* | Yes | 6540.99 | 6540.97 | −0.02 | −0.00 % | — |

`approx_mode` Yes and No measure identically for all six ops — none of these kernels branches
on it — so the table lists each row once.

Reading the table:

- **SqrtCustom is the worst relative cost and the smallest absolute one.** The same +96 cycles
  land as +9.7 % here and as +4.0 % on `Asin`/`Acos`, because `sqrt_custom` is a cheap
  ~984-cycle kernel and the trig ones are ~2.4k. Nothing about the guard is more expensive on
  the direct op.
- **`Asin` and `Acos` at `dest_acc=Yes` do not move at all** — exactly 0.00, not "within
  noise". Those variants never instantiate `sqrt_custom`; §6 confirms it from the ELF.
- **The controls are flat**, which is what makes the other rows attributable to this change
  rather than to the two trees being built differently.
- Other run types: `UNPACK_ISOLATE` is 0.00 % on every variant. `L1_TO_L1` tracks `MATH_ISOLATE`
  (+3.9 % to +9.7 % on the affected rows) because math is the bottleneck for these ops.

## 6. Where the cycles go

Instruction-count delta of the `MATH_ISOLATE` `math.elf`, paired by build hash:

| Op | Wormhole B0 | Blackhole |
|---|---|---|
| `sqrt_custom` | +6 | +6 |
| `acos` | +6 / 0 | +6 / 0 |
| `asin` | +6 / 0 | +6 / 0 |
| `erfinv` | +4 | +3 |
| `acosh` *(control)* | 0 | 0 |
| `asinh` *(control)* | 0 | 0 |

The `+6 / 0` split for `asin`/`acos` is exactly the `dest_acc` split in §5: the variants that
gain 6 instructions are the ones that gain 96 cycles, and the ones that gain none gain none.

For `sqrt_custom` the six are, by SFPU mnemonic histogram:

```
+2 sfpexexp     extract the biased exponent
+2 sfpiadd      compare it against 255
+2 sfpsetcc     combine with the existing != 0.0f predicate
```

Two of each because the kernel's loop is unrolled ×2 — **3 instructions per SFPU vector
iteration**. A tile is 32 vector iterations, so 32 × 3 = **96 cycles per tile against 96.01
measured**. The cost is exactly the instructions added, at one cycle each, with nothing
attributable to scheduling or register pressure.

This is also the evidence that the guard is tested on the exponent field rather than as a
compare against infinity: `SFPSETCC` is specified only for inputs that are not NaN
(`VectorUnit.md`), and this predicate is evaluated on a possible NaN. The exponent form costs
the same three instructions and has no such caveat.

No register-allocator ICE on either arch.

## 7. Suite results on this branch

Wormhole n300, this branch, `test_sfpu_unary.py`:

| Sweep | Result |
|---|---|
| Edge sweep (`-k edges`) | **487 passed, 27 xfailed, 0 xpassed, 0 failed**, 238 skipped |
| Consumer sweep (`-k "Erfinv or SqrtCustom or Asin or Acos or Sqrt or Rsqrt"`) | **1053 passed, 17 xfailed, 0 xpassed, 0 failed**, 2 skipped |

Baseline edge sweep for comparison: 491 passed, 23 xfailed, 0 xpassed, 0 failed. The −4
passed / +4 xfailed is two effects, not a regression:

- `Erfinv`'s **2** xfails are gone — those cells now assert `erfinv(±1) == ±inf` instead of
  pinning NaN. (+2 passed, −2 xfailed)
- Enrolling `SqrtCustom` in `SPECIALS_READY_OPS` adds **6** xfails for `sqrt_custom(-inf)`.
  (−6 passed, +6 xfailed)

Those 6 cells were not passing *correctly* before: `SqrtCustom` was absent from the table, so
the non-finite probe was never sent at them. That absence is why `+inf` was never driven at
the op and the defect had to be found through `erfinv` — the real lesson of the finding. The
6 xfails are the cost of making the op honest about a limit it already had.

These figures are specific to this branch. The development-era commit message quotes 495
passed / 19 xfailed, measured when the reciprocal-pole fix was on the same branch; that fix
now lives on `ldjurovic/sfpu_52930_reciprocal_compat_pole` and its 8 `RsqrtCompat` xfails are
therefore still present here. 487 + 8 = 495 and 27 − 8 = 19, so the two runs agree.

## 8. What was not measured

| # | Item | Why |
|---|---|---|
| 1 | Blackhole cycle counts | No BH silicon on this host. BH is compile-verified (328 correctness variants, 24 perf variants, clean) and its instruction delta is the same shape as Wormhole's — but a cycle figure would be a guess. |
| 2 | Quasar | Out of scope for this PR; the kernel is untouched there. It carries the same defect. |
| 3 | `Sqrt` / `Rsqrt` | Different kernel (`_calculate_sqrt_body_`), not a `sqrt_custom` consumer. In the consumer sweep as passengers only. |
| 4 | End-to-end model impact | `sqrt_custom` has no perf instrumentation above the LLK layer here. +9.7 % on a 984-cycle kernel is the LLK-level cost, not a model-level one. |
