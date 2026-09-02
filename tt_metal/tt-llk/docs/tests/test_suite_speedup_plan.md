# LLK test-suite speedup plan

Everything below is measured on this repo, not estimated from reading code. Where a
number is an extrapolation it says so.

**Measurement setup.** `tt-metal` @ `effd2e1477b`, Blackhole p150 (`/dev/tenstorrent/0`),
`CHIP_ARCH=blackhole`, 6 CPU cores, Python 3.10.19 / torch 2.9.1+cpu / numpy 2.2.6,
sfpi 7.72.0. All sweep runs are **serial** (`-n 1`) so the stage split is not confounded
by xdist; CI runs `-n auto --dist=worksteal` over 2 `pytest-split` groups.

---

## 1. Baseline

### Suite size (Blackhole, `-m "not perf and not quasar and not accuracy"` — what llk-e2e runs)

| | tests | share |
|---|---:|---:|
| **whole e2e collection** | **237,739** | 100% |
| `test_math_matmul.py` | 150,360 | 63.2% |
| `test_unpack_matmul.py` | 36,416 | 15.3% |
| `test_matmul_custom.py` + `test_matmul.py` | 13,664 | 5.7% |
| **all `*sfpu*` files together** | **11,325** | **4.8%** |
| — `test_eltwise_unary_sfpu.py` | 6,662 | 2.8% |
| — `test_eltwise_binary_sfpu.py` | 1,878 | 0.8% |
| — `test_sfpu_reduce.py` | 1,622 | 0.7% |
| everything else | 25,974 | 10.9% |

SFPU is 4.8% of the *count* but its tests are individually expensive, so it is a much
larger share of wall time than the count suggests. `test_math_matmul` is the elephant.

### The reference workload: `test_eltwise_unary_sfpu::test_eltwise_unary_sfpu`

5,792 collected → **4,652 executed, 1,140 skipped (19.7%)**, 570 unique ELFs.

| run | wall |
|---|---:|
| 1st (cold artefact dir) | **661 s** |
| 2nd, same `RUNNER_TEMP` | **658 s** |

**The second run is not faster.** `TestConfig` wipes the artefact directory at session
start whenever `BUILD_MODE` is `DEFAULT` or `PRODUCE`
([`test_config.py:881-887`](../../tests/python_tests/helpers/test_config.py#L881-L887)),
so a plain `pytest` invocation — which is what the entire e2e lane and every local run
uses — recompiles all 570 ELFs every time. Within a single session the cache works
(4,652 tests share 570 ELFs, ~8:1); across sessions there is no reuse at all.

---

## 2. Where the time goes

Instrumented run of the full unary sweep — every stage wrapped in a `perf_counter`
accumulator loaded as a pytest plugin (no cProfile, so no bias toward the pure-Python
stages). 4,652 executed tests, **805.5 s of 819 s wall accounted for = 98.3%**, so nothing
significant sits outside this table.

(This run's wall is 819 s against §1's 661 s baseline — run-to-run variance on a shared
6-core box, not instrumentation cost: the wrappers add 8 calls per test. Use the *shares*
below, applied to the 661 s baseline, which is what §4 does. The two reconcile on the
compile stage, which can be checked independently: two smaller samples measured 438 ms and
423 ms per unique ELF, and 570 ELFs × ~440 ms = ~250 s = 38% of 661 s, matching the 38.5%
share. The instrumented run's compiles averaged 544 ms each under heavier load, which is
where its extra ~150 s went.)

| stage | seconds | ms / test | share |
|---|---:|---:|---:|
| **compile** (`build_elfs`) | 310.0 | 66.6 | **38.5%** |
| **golden** (`UnarySFPUGolden.__call__`) | 228.6 | 49.1 | **28.4%** |
| **stimuli pack + L1 write** (`StimuliConfig.write`) | 189.0 | 40.6 | **23.5%** |
| result read + unpack (`collect_results`) | 55.8 | 12.0 | 6.9% |
| stimuli generation (`generate_stimuli`) | 13.4 | 2.9 | 1.7% |
| ELF load + launch (`run_elf_files`) | 4.4 | 0.9 | 0.5% |
| compare vs golden (`passed_test`) | 4.1 | 0.9 | 0.5% |
| device wait (`wait_for_tensix_operations_finished`) | 0.2 | 0.05 | 0.0% |
| *accounted* | *805.5* | | *98.3%* |

**The most important row is the bottom one: the device is 0.5% of this sweep.** Waiting on
Tensix is 0.2 seconds out of 819, and launching the ELFs another 4.4. Everything else is
host-side Python and `g++`. Any plan that tries to buy speed by touching the device path,
the NOC transfers or the worker count is optimising 1% of the problem.

Two drill-downs, from a cProfile run of the same sweep (shares there are skewed toward
Python-heavy stages, but the *attribution within* a stage is exact):

- Of the 189 s stimuli stage, **87% is `float_to_bfp8_block`** — 4,534,272 calls, driven by
  9,304 `write_matrix` calls for 4,652 tests (exactly 2× — see idea F).
- Of the 228.6 s golden stage, essentially all of it is the per-element listcomp, with
  **28,958,720 `_torch_unary` calls** underneath it.

`build_elfs` is ~100% real compilation, not bookkeeping: `generate_build_header` measures
0.01 ms/call and `build_shared_artefacts` 0.42 ms/call, and the 4,082 calls that hit the
in-session cache cost essentially nothing. The 570 actual compiles account for the whole
stage.

### Per-test cost across the suite

| workload | tests | ms / test | unique ELFs |
|---|---:|---:|---:|
| `test_eltwise_unary_sfpu` (main sweep) | 4,652 executed | 142 | 570 (8:1) |
| `test_math_matmul` (188-case stride sample) | 188 | 84 | 25 (7.5:1) |

At those rates the whole 237,739-test e2e collection is **roughly 5–6 hours of serial
work**, which the two `pytest-split` groups × `-n auto` have to fit into a 38-minute
timeout each.

---

## 3. The ideas, ranked

Ranking is by (projected saving) / (effort × risk). Each item says what evidence backs
the projection, so a reviewer can disagree with the number rather than the vibe.

### A — Stop recompiling every ELF on every run  ⭐ biggest single win

**What.** The per-variant ELF cache already exists and already works *within* a session:
`build_elfs` short-circuits on a `.build_complete` marker under
`ARTEFACTS_DIR/<test>/<variant_id>/`, which is why 4,652 tests need only 570 compiles. The
only reason it gains nothing *across* sessions is the unconditional
`shutil.rmtree(ARTEFACTS_DIR)` at session start.

**Why the wipe is there.** It is a *correctness guard*, not an oversight. `variant_id` is a
sha256 of the resolved compile/link option string plus the include search dirs
([`test_config.py:1299-1301`](../../tests/python_tests/helpers/test_config.py#L1299-L1301)) —
so it captures the flags and the paths, but **not the contents of the headers those paths
point at**. Edit `llk_math_eltwise_unary_sfpu.h` and every variant id is unchanged, so a
surviving cache would silently serve stale ELFs. That is why the fix is not "delete the
rmtree".

**The actual fix.** Make the cache key sound, then stop wiping:
1. Fingerprint the compile *inputs* — a content hash over the LLK header trees,
   `tests/sources/`, `tests/helpers/` and the sfpi version — into one `TOOLCHAIN_KEY`.
2. Put `TOOLCHAIN_KEY` in the artefact path
   (`ARTEFACTS_DIR/<toolchain_key>/<test>/<variant_id>/`), so a source edit lands in a
   fresh namespace instead of having to invalidate entries in place.
3. Wipe only on an explicit `--rebuild`, and garbage-collect old key namespaces by age.

A plain content hash of those trees is a good enough first cut. The precise version is
`g++ -M` dependency output per variant, which is strictly better but more plumbing; start
with the coarse key, since it errs toward over-invalidation rather than under.

**Projected.** Removes the whole compile stage from every repeat run — see §4. Applies to
the *entire* suite, not just SFPU. In CI it additionally needs an `actions/cache` keyed on
`TOOLCHAIN_KEY`; hit rate then tracks how often LLK sources change, which for a
PR-gate/nightly split is most of the time.

**Effort** medium. **Risk** medium — a wrong key serves stale ELFs, which is exactly the
failure mode the current wipe prevents. Mitigate by landing the fingerprint first, behind
a flag, and diffing ELF hashes against a wiped run for one full sweep.

### B — Give the e2e lane the producer/consumer split the perf lane already has

**What.** The perf lane and the ttsim regression already do this:

```
pytest --compile-producer -n 10  -m "perf and not accuracy" ...   # compile only, no device
pytest --compile-consumer -n 15  -m "perf and not accuracy" ...   # execute only
```
(`tests/run_llk_perf_blackhole.sh:38-41`, `run_ttsim_regression.sh:322`)

The e2e lane does not — `tests/pipeline_reorg/llk_e2e_tests.yaml` runs plain
`pytest -n auto`, so every worker interleaves `g++` with holding a Tensix core.

**Why it helps even without A.** Compile is pure CPU with no device dependency, so the
producer pass can run at a much higher `-n` than the device pass (the perf lane uses 10
vs 15 for exactly this reason) and, better, on a CPU-only runner *before* the device
runner is allocated. `_collapse_runtime_only_variants`
([`llk_pytest_plugin.py:565-593`](../../tests/python_tests/helpers/llk_pytest_plugin.py#L565-L593))
already dedupes the producer pass down to one item per compile key.

**Projected.** Moves the compile stage off the device-runner critical path. Combined with
A it is close to free on repeat runs.

**Effort** small — the flags exist and are proven in-tree. **Risk** low.

### C — Collapse the `throttle` 1–5 axis in `test_math_matmul`  ⭐ biggest count reduction

**What.** `test_math_matmul` is 150,360 tests. The throttle axis accounts for almost all of it:

| throttle | tests |
|---:|---:|
| 0 (tiny tiles only) | 11,320 |
| 1 | 27,808 |
| 2 | 27,808 |
| 3 | 27,808 |
| 4 | 27,808 |
| 5 | 27,808 |

`throttle` is consumed in exactly one place — `THROTTLE_LEVEL(throttle)` in the compile
templates ([`test_math_matmul.py:220`](../../tests/python_tests/test_math_matmul.py#L220)).
The stimuli, the golden and the assertion never see it, so all five levels assert the
*same expected value on the same inputs*. On the kernel side the levels differ only in how
many NOPs are interleaved between MVMULs to cap throughput — 1→73%, 2→67%, 3→50%, 4→40%,
5→33% (`tt_llk_blackhole/llk_lib/llk_math_matmul.h:547`) — plus the per-level replay-buffer
length.

**The proposal.** Keep one representative non-zero level across the full
format × dims × fidelity matrix, and add a *dedicated* throttle sweep that covers all five
levels over a small slice (all 5 levels × 4 fidelities × a handful of format/dim points ≈
a few hundred tests). That preserves the coverage that actually differs per level — the
`replay_buff_len_throttle` bookkeeping, where a mismatch would corrupt results — without
multiplying it by 27,808.

**Projected.** ~111,000 tests removed = **47% of the entire e2e collection**. And because
throttle is a *compile-time* parameter, it also cuts `test_math_matmul`'s ELF count by
~5×, and compile is the expensive half under §2's numbers.

**Effort** small (a change to `ALL_TEST_PARAMS`). **Risk** medium, and it is a **coverage
decision, not a refactor** — it needs the matmul owner's sign-off, not a unilateral edit.
Level 0 vs non-zero is a genuine structural difference (a different MOP and an extra
fidelity-clear addrmod) and must stay fully covered.

### D — Vectorise `UnarySFPUGolden`  ⭐ the SFPU-specific win

**What.** The golden for every unary SFPU op is computed **one scalar at a time in
Python**:

```python
# golden_generators.py:2582
op_res = [self.ops[operation](x) for x in result.tolist()[window]]
```

and 33 of those ops route through `_torch_unary`
([`golden_generators.py:2702`](../../tests/python_tests/helpers/golden_generators.py#L2702)),
which builds a **0-d `torch.tensor` per element** and calls `.item()`. Over the sweep that
is 28,958,720 `_torch_unary` calls, 51,174,704 `torch.tensor` constructions and 50,893,356
`.item()` calls. `EltwiseBinaryGolden`, `MatmulGolden` and the
rest of the golden layer are already fully vectorised — `UnarySFPUGolden` is the outlier.

**Prototype, validated.** The replacement is three lines:

```python
def torch_unary_vec(t, torch_fn, is_exponent_b):
    r = torch_fn(t.float())
    if not is_exponent_b:                     # A-exponent dest: inf -> NaN
        r = torch.where(r.isinf(), torch.full_like(r, math.nan), r)
    return r
```

Checked **bit-exact** (NaN-payload-insensitive) against the scalar path for
sqrt/rsqrt/log/exp/sin/erf/tanh × {exponent_b, not exponent_b} on a 32,768-element
population seeded with ±0, ±inf, NaN, subnormals and format-max values — **0 differing
bits in all 14 combinations**. Speedups at 32,768 elements: 10× (exp/erf/tanh/log) to
1,700× (sqrt/rsqrt).

**Effort** medium — the registry has 118 ops; the 33 `_torch_unary` ones convert
mechanically, the rest need case-by-case review (some are `math.*` predicates that
vectorise trivially, a few are genuinely elementwise-irregular). **Risk** low: the
existing scalar path is a perfect oracle, so each op can be proven bit-identical before
the loop is deleted. Do it op-family at a time behind a shared helper.

Also applies to `_call_integer`
([`golden_generators.py:3026`](../../tests/python_tests/helpers/golden_generators.py#L3026)),
the same pattern on the integer SFPU path.

### E — Vectorise the BFP pack/unpack helpers  ⭐ suite-wide

**What.** `float_to_bfp8_block` ([`pack.py:150`](../../tests/python_tests/helpers/pack.py#L150))
does bit manipulation by formatting each value as a **binary string**:

```python
def bfloat16_to_binary(value):
    float_value = struct.unpack("<I", struct.pack("<f", value))[0]
    bfloat16_value = (float_value & 0xFFFF0000) >> 16
    return f"{(bfloat16_value >> 8) & 0xFF:08b}{bfloat16_value & 0xFF:08b}"
```

then slices the string and `int(s, 2)`s it back, per element. Over the sweep: 4,534,272
calls to `float_to_bfp8_block`, and beneath them **72,548,352 calls to
`bfloat16_to_binary`** and **88,486,552 `struct.pack` calls**. It also iterates a torch
tensor element-wise (`for value in block`), which shows up as 4,568,272 `Tensor.unbind`
calls costing 51 s on their own.

**Prototype, validated.** A numpy version is in §8. Verified **byte-identical** — both shared
exponents and all mantissas — against `float_to_bfp8_block` on three populations
(uniform, wide-dynamic-range, all-subnormal), each seeded with ±0/±inf/NaN/1e30/1e-30.
**58×–68× faster.**

One subtlety the prototype had to reproduce rather than "fix": `binary_str[9:-1]` keeps
bits 9–14 of the 16-bit word, i.e. it **drops the bf16 mantissa LSB**, then prepends the
implicit 1 for a *7-bit* explicit mantissa. The obvious `(bf16 & 0x7F) | 0x80` is 8 bits
and does not match. That is why the byte-comparison test matters.

**Scope.** Every test in the suite with a Bfp8_b / Bfp4_b / Bfp2_b operand or result, not
just SFPU. The mirror-image `_bfp_to_float_block`
([`unpack.py:125`](../../tests/python_tests/helpers/unpack.py#L125)) still loops per datum
with a memo dict and takes the same treatment.

**Effort** small-medium. **Risk** low — byte-comparable against the current implementation.

### F — Stop generating, packing and DMA-ing an operand the unary kernels never read

**What.** Every unary SFPU test asks `generate_stimuli` for a full `src_B`
([`test_eltwise_unary_sfpu.py:1140-1146`](../../tests/python_tests/test_eltwise_unary_sfpu.py#L1140-L1146))
and hands it to `StimuliConfig`, which unconditionally packs it and writes it to L1
([`stimuli_config.py:664`](../../tests/python_tests/helpers/stimuli_config.py#L664)).
`sources/eltwise_unary_sfpu_test.cpp` contains **zero references to `buffer_B`**. The
profile shows `write_matrix` called 9,304 times for 4,652 tests — exactly 2× — so half of
the stimuli-write stage is spent on an operand no kernel reads.

**Fix.** `StimuliConfig` already has the machinery: `_OPTIONAL_OPERAND_SPECS` makes S/T/C
optional. Move B into it (keeping the L1 address reservation, which the generated
`params.h` declares) and let the unary driver omit it.

**Projected.** Half the stimuli-write stage for the unary sweep, plus one fewer device
write per test. Largely subsumed by E — once packing is ~free, only the DMA remains — but
worth doing on its own for clarity.

**Effort** small. **Risk** low.

### G — Balance the CI split by duration, not by test count

**What.** CI runs `pytest-split --splits 2 --group N`, and there is **no `.test_durations`
file anywhere in the repo** and no `--store-durations` in any workflow. pytest-split's own
fallback is explicit about what that means:

```
[pytest-split] No test durations found. Pytest-split will split tests evenly
when no durations are found.
```
(`pytest_split/plugin.py:143-146`)

So the two groups are balanced by *count*, across a suite whose per-test cost spans more
than an order of magnitude (cheap `test_math_matmul` cases vs `[128,256]` Bfp8_b SFPU
cases). One group approaches the 38-minute timeout while the other idles.

**Fix.** `--store-durations` on one scheduled run, commit `.test_durations`, refresh it
periodically. Nothing else changes.

**Projected.** Recovers up to half the current imbalance on the critical path.
**Not yet quantified** — it needs one instrumented CI run to measure the actual skew, and
that is the first thing to do here.

**Effort** trivial. **Risk** none.

### H — Filter unconditional skips at parametrize time instead of in the test body

**What.** 1,140 of 5,792 unary-sweep cases (19.7%) skip on Blackhole. 248 of those skip
*unconditionally on every arch*:

- `ReluMin` — 168 cases, `pytest.skip` at
  [`test_eltwise_unary_sfpu.py:437`](../../tests/python_tests/test_eltwise_unary_sfpu.py#L437)
  (blocked on tt-llk#1120), yet the op is listed in `BROAD_SWEEP_OPS` and generates its
  whole matrix.
- `Tanh` + `ApproximationMode.Yes` — 80 cases, skipped at line 441.

The remaining ~890 are the Blackhole format guards (`_skip_bh_unless_fp32` /
`_skip_bh_unsupported_float_combo`). Those are equally movable, because
`TestConfig.CHIP_ARCH` is bound in `pytest_configure` — i.e. before collection — so the
format lists can be filtered when the params are built rather than rejected per test.

**Projected.** Near zero in wall time; a skip is cheap. The value is that the two together
remove ~20% of the collected count, which shrinks the xdist work queue and the JUnit
report, and stops a 20% skip rate from sitting between you and the real coverage number.
Treat this as hygiene that makes the other measurements honest, not as speed.

**Effort** small. **Risk** low.

---

## 4. Projections

> **Status: D, E, F and H are implemented and measured.** The projections in this section
> are the *pre-implementation* estimates, kept as written so they can be compared against
> what actually happened. The measured outcome is below; where the two disagree, the
> measured numbers are the ones to trust.
>
> ### Measured outcome (D + E + F + H)
>
> Back-to-back runs of `test_eltwise_unary_sfpu::test_eltwise_unary_sfpu` on the p150,
> same machine state (compile 233.9 s before vs 234.1 s after confirms comparability):
>
> | | before | after |
> |---|---:|---:|
> | wall | 609 s | **364 s** (−40%, 1.67x) |
> | non-compile work | 364.8 s | **121.2 s** (3.0x) |
> | outcome | 4652 passed, 1140 skipped | 4652 passed, **0 skipped** |
>
> | stage | before ms/test | after ms/test | factor |
> |---|---:|---:|---:|
> | compile (untouched) | 50.28 | 50.32 | 1.0x |
> | golden (D) | 36.24 | 13.61 | **2.7x** |
> | stimuli pack + write (E, F) | 29.69 | 0.82 | **36x** |
> | stimuli generation (F) | 2.00 | 1.23 | 1.6x |
> | result read + unpack | 9.05 | 9.03 | 1.0x |
>
> Where the estimates were wrong, and why:
>
> - **D came in at 2.7x, not the projected ~15x.** The projection assumed every op could
>   be vectorised. 26 of 114 cannot: 18 because torch selects a different kernel for a
>   1024-element tensor than for the 0-d tensor the scalar path builds and the answers
>   differ in the last fp32 bit, and 8 because they are not elementwise-unary-float ops
>   at all. Forcing all 18 through torch anyway *was* tried: it reaches 2.34 ms/test
>   (15.5x) and the main sweep passes, but it fails
>   `test_eltwise_unary_sfpu_edges[Float32->Float32, GeluAppx, dest_acc=Yes]` — on an
>   fp32 Dest there is no 16-bit rounding to absorb the 1-ULP shift. Reverted.
> - **E beat its projection** (36x on the stage vs the estimated ~58x on the function
>   alone, which is the same thing measured end-to-end including the DMA that remains).
> - **F is now worth almost nothing on its own**, exactly as §4 predicted once E lands:
>   the packing it avoided is already free.
> - **H was projected as ~0 wall and delivered ~0 wall.** Its value is that the sweep now
>   reports 4652/0 instead of 4652/1140, and collection dropped 5792 -> 4652 — exactly
>   the number that used to pass, which is the invariant proving nothing was lost.
>
> The 1-ULP exclusions are enforced by
> `test_unary_sfpu_golden_vectorised.py`, and the BFP packer by
> `test_bfp_pack_vectorised.py`; both keep the per-element implementation as the oracle
> rather than deleting it.


Savings are expressed against the **661 s un-instrumented baseline** for
`test_eltwise_unary_sfpu::test_eltwise_unary_sfpu`, using the §2 stage shares. Speedup
factors are the measured prototype numbers; where a factor is assumed conservatively the
table says so.

E and F both attack the stimuli stage, so their standalone savings overlap and must not be
added. The cumulative figures below apply E first (which makes packing nearly free) and
then count only F's residual — the DMA write itself.

### On the reference SFPU sweep (serial, 661 s baseline)

| idea | stage attacked | stage cost | after | saved | Δ sweep | confidence |
|---|---|---:|---:|---:|---:|---|
| **D** vectorise `UnarySFPUGolden` | golden | 188 s | ~13 s | 175 s | **−26%** | high — prototype bit-exact, ≥15× assumed vs 10–1700× measured |
| **E** vectorise BFP pack | 87% of stimuli write | 135 s | ~2 s | 133 s | **−20%** | high — prototype byte-identical, 58–68× measured |
| **F** drop the unused `buffer_B` | operand B's half of stimuli write | 155 s | 77 s | 77 s | **−12%** alone, **−1.5%** if E lands first | high — kernel provably never reads it |
| **A** sound artefact cache | compile, on repeat runs | 254 s | 0 s | 254 s | **−38%** | high on the saving, medium on the key being sound |
| **B** producer/consumer split | compile, off the device pass | 254 s | moved | — | −38% of *device-runner* wall | high |
| **H** drop unconditional skips | collection only | — | — | ~0 s | ~0% | hygiene, not speed |
| E's read-side twin (`_bfp_to_float_block`) | part of result read | ≤46 s | — | unmeasured | ≤−7% | medium — same pattern, not yet profiled apart |

**Cumulative, cold single run** (compile still paid inline):
661 s → 343 s, **−48%**.

**Cumulative, repeat run or with a sound artefact cache** (A landed):
661 s → **~90 s, −86% (7.3×)** — and none of that comes from doing less testing or less
device work. Composition of the remaining 90 s: result read 46 s, golden 13 s, stimuli
10 s, generation 13 s, launch/compare 5 s, compile 0 s.

That residual is dominated by the result-read path, which is the natural next target once
D/E/F land — the same numpy treatment applied to `unpack_res_tiles` / `_bfp_to_float_block`.

### Across the whole suite

| idea | mechanism | scope | projected |
|---|---|---|---|
| **C** collapse `throttle` 1–5 | removes ~111,000 tests *and* ~4/5 of `test_math_matmul`'s ELFs | `test_math_matmul` | **−47% of the entire e2e collection**; at the measured 84 ms/test that is ~2.6 h of serial work |
| **E** vectorise BFP pack | every test with a Bfp8_b/Bfp4_b/Bfp2_b operand or result | suite-wide | proportional to each test's BFP share; 87% of the stimuli stage where it applies |
| **A** + **B** artefact cache & producer split | compile is 38.5% of a sweep and ~44% of the math_matmul sample | suite-wide | up to −40% of repeat-run wall time |
| **G** duration-balanced split | removes group skew | CI only | unquantified — measure first |
| **D** vectorise `UnarySFPUGolden` | unary SFPU goldens only | 4.8% of the suite by count, more by time | −26% of the SFPU sweeps |

**Order-of-magnitude summary.** C is the only idea that changes *how many* tests run, and
it is worth about as much as everything else combined. A/B/D/E/F change only *how fast the
same tests run*, and together they are worth roughly a 2× on a cold run and a 7× on a
repeat run of the SFPU sweeps. Nothing in this plan touches device time, because device
time is 0.5%.

---

## 5. Deliberately not recommended

### The half-built golden/stimuli disk cache

`--stimuli-only` / `--use-stimuli` exist and work: they flip `GeneratorProxy.MODE`, swap
`get_golden_generator` for `get_golden_proxied`, and load `golden.pt` from disk instead of
computing it ([`test_config.py:855-877`](../../tests/python_tests/helpers/test_config.py#L855-L877),
[`golden_generators.py:479-565`](../../tests/python_tests/helpers/golden_generators.py#L479-L565)).

Do not build on it:

- **No workflow or script uses either flag.** `GeneratorProxy.MODE` is assigned nowhere
  else, so `get_golden_proxied` raises `ValueError("GeneratorProxy mode not set")` if it is
  ever reached by another path.
- **The cache key is a sha256 of `PYTEST_CURRENT_TEST`** — the test *name*. Rename a
  parametrize id and every entry silently misses; change a test *body* or a golden's maths
  and every entry silently **hits** and serves a stale golden. That is the wrong direction
  of failure for a correctness oracle.
- **It only half-works anyway**: `load_from_cache()` runs *after* the test body has already
  called `generate_stimuli`, so the stimuli are generated and then overwritten. Only the
  golden cost is actually avoided.
- D and E make it pointless. Once the golden is vectorised it is faster to recompute than
  to `torch.load` it.

Either wire it up deliberately with a content-addressed key, or delete it. Leaving a dark
caching path next to a correctness oracle is a trap.

### Trimming the SFPU format or op matrix

Tempting because `test_eltwise_unary_sfpu` looks huge, but it is 2.8% of the suite and the
matrix is where the real bugs live — the broad/standard profile split
([`test_eltwise_unary_sfpu.py:63-84`](../../tests/python_tests/test_eltwise_unary_sfpu.py#L63-L84))
is already a deliberate cost/coverage tradeoff, and the file documents measured
format-specific divergences (approximate exp's rtol overshoot, the signed-zero
`unpack_to_dest` partition) that only exist because the matrix is wide. Spend the effort on
D/E/F, which make the same coverage cheaper, rather than on buying speed with coverage.

### Raising `-n` on the device pass

The device is not the bottleneck — `run_elf_files` plus the device wait is a low
single-digit percentage of the sweep (§2). Each xdist worker already gets its own Tensix
core ([`test_config.py:825-836`](../../tests/python_tests/helpers/test_config.py#L825-L836)),
so the ceiling is host CPU, not Tensix. A wider CI runner does buy real parallelism — but
it buys it against a workload that is 98% Python and `g++`, so the lever that matters is
still reducing that work (D, E, F, A), not adding workers on top of it. Worth noting the
corollary: every one of D/E/F makes each xdist worker cheaper, which raises the effective
`-n` a given runner can sustain.

---

## 6. Suggested order

1. **G** — one instrumented CI run to measure the split skew, then commit `.test_durations`.
   Trivial, zero risk, and it tells you how much of the current wall time is just imbalance.
2. **E** then **D** — the two vectorisations. Both have validated bit-exact prototypes and
   the current code is its own oracle. E first: it is smaller, suite-wide, and the
   byte-comparison harness it needs is reusable for D.
3. **F**, **H** — small, local, ride along with the SFPU work.
4. **B** — producer/consumer for the e2e lane. Proven pattern already in-tree.
5. **A** — the sound artefact cache. Highest payoff, but it must land *after* B, because
   the producer/consumer split is what makes a persistent artefact namespace useful, and it
   needs the ELF-hash diff harness to be trustworthy.
6. **C** — the throttle axis. Land last, not because it is hard, but because it is the one
   item that trades coverage for time and therefore needs a decision from the matmul owner
   rather than a patch.

Steps 1–5 are pure speed with no coverage change. Step 6 is the only one that changes what
is tested — keep it separable so it can be reverted independently.

---

## 7. Reproducing the measurements

```bash
cd tt_metal/tt-llk/tests/python_tests
export CHIP_ARCH=blackhole
export RUNNER_TEMP=$(mktemp -d)          # isolated artefact root per variant

# suite size
pytest --collect-only -q . -m "not perf and not quasar and not accuracy" | tail -1

# reference workload, serial
time pytest -q --no-header --override-ini=log_cli=false \
    test_eltwise_unary_sfpu.py::test_eltwise_unary_sfpu

# unique ELFs actually built
find "$RUNNER_TEMP" -name .build_complete | wc -l

# per-stage wall time: wrap build_elfs / StimuliConfig.write / collect_results /
# UnarySFPUGolden.__call__ / generate_stimuli in a perf_counter accumulator loaded as a
# pytest plugin. cProfile also works but inflates the pure-Python stages by ~35% relative
# to the subprocess-bound compile stage, so it is fine for ranking and wrong for shares.
```

Do **not** measure with a reused `RUNNER_TEMP` and assume the build is warm: the artefact
wipe described in §1 means it never is.

---

## 8. Appendix — the validated prototypes

Both are drop-in replacements for the current hot loops and were checked against the
existing code as the oracle. Neither is committed anywhere; they are here so the
projections in §4 can be re-derived rather than trusted.

### D — `UnarySFPUGolden._torch_unary`, vectorised

```python
def torch_unary_vec(t: torch.Tensor, torch_fn, is_exponent_b: bool) -> torch.Tensor:
    r = torch_fn(t.float())
    if not is_exponent_b:                      # A-exponent dest: inf -> NaN
        r = torch.where(r.isinf(), torch.full_like(r, math.nan), r)
    return r
```

Validation: bit-exact (NaN-payload-insensitive) vs the scalar path for
sqrt / rsqrt / log / exp / sin / erf / tanh × {exponent_b, not exponent_b} on a
32,768-element population seeded with ±0, ±inf, NaN, subnormals and format-max values —
**0 differing bits in 14/14 combinations**. 10×–1700×.

### E — `float_to_bfp8_block`, vectorised

```python
def bfp8_blocks_vec(t: torch.Tensor):
    x = t.to(torch.float32).contiguous()
    u = x.view(torch.int32).numpy().astype(np.uint32)
    bf16 = (u >> 16) & 0xFFFF
    s = (bf16 >> 15) & 1
    e = (bf16 >> 7) & 0xFF
    # The scalar path keeps bits 9..14 of the 16-bit word -- i.e. it DROPS the bf16
    # mantissa LSB ("remove last") -- then prepends the implicit 1, giving a 7-bit
    # explicit mantissa. Reproduce exactly; do NOT "fix" this to 8 bits.
    m = ((bf16 >> 1) & 0x3F) | 0x40
    m = m.reshape(-1, 16); e = e.reshape(-1, 16); s = s.reshape(-1, 16)
    shared = e.max(axis=1, keepdims=True)
    d = (shared - e).astype(np.int64)
    guard = np.where(d > 0, (m >> np.maximum(d - 1, 0)) & 1, 0)     # RNE, ties away
    shifted = np.where(d > 0, (m >> np.minimum(d, 63)) + guard, m)
    mag = (shifted & 0x7F).astype(np.uint32)
    out = np.where(mag != 0, (s << 7) | mag, 0).astype(np.uint8)    # flush -0 to +0
    return shared.reshape(-1).astype(int).tolist(), out.reshape(-1).tolist()
```

Validation: byte-identical shared exponents **and** mantissas vs `float_to_bfp8_block` on
three populations (uniform, wide-dynamic-range, all-subnormal), each seeded with
±0 / ±inf / NaN / 1e30 / 1e-30. **58×–68×.**

The mantissa-LSB subtlety is the whole reason this needs a byte-comparison test and not a
value-comparison test: the "obvious" `(bf16 & 0x7F) | 0x80` passes a tolerance check and
produces different bytes.

### Suggested landing pattern for both

1. Add the vectorised function next to the scalar one.
2. Add a test that asserts the two agree **bitwise** over a seeded population that includes
   the specials, parametrised over every op / format the scalar path supports.
3. Switch the caller.
4. Delete the scalar path in a follow-up, once the agreement test has run in nightly.

Step 2 is the deliverable that makes steps 3 and 4 safe, and it is reusable for the
`_call_integer` and `_bfp_to_float_block` follow-ups.
