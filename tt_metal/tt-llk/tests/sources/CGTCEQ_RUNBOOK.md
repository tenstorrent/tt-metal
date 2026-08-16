# CGTCEQ bench RUNBOOK — (Cgt,Ceq) exact-count engine microbenchmark

**Scope / honesty guard:** this bench prices the **Gate-2 correctness ORACLE
only** (RADIX_BUCKET_GPU.md IMPL-2 consensus). Even a perfect 25-cyc rendezvous
creates no win region before Gate 4; the outputs are measured constants for
RADIX_BUCKET_GPU.md correction #8 (rendezvous 25–100 cyc prior) and dep-map
open dep #1, plus an honest SFPU-side comparator for the Gate-3 shootout.

**Files (all NEW, no tracked file edited):**

- `tt_metal/tt-llk/tests/sources/cgtceq_perf.cpp` — the kernel (8 arms).
- `tt_metal/tt-llk/tests/python_tests/test_cgtceq_perf.py` — driver + exact
  goldens (named `test_*` deliberately: the `perf_*.py` schema merge gate
  derives its catalog from `perf_*.py` files only, so no
  `helpers/perf/test_schemas.py` edit is needed).
- `tt_metal/tt-llk/tests/python_tests/cgtceq_analysis.py` — host-only
  post-processing (additivity table, 3×3 rendezvous matrix, bisect p50/p95).

## Prerequisites

- Blackhole silicon (p150a). The kernel is BH-only (SFPGT SET_VD,
  SFPLOADMACRO). ttsim CANNOT substitute (no SFPLOADMACRO, not cycle-accurate).
- Run everything from `tt_metal/tt-llk/tests` with its venv:

```bash
cd /home/nachiket/tt-metal/tt_metal/tt-llk/tests
source .venv/bin/activate
export CHIP_ARCH=blackhole
```

- ttexalens must import (it does in this venv; the conftest hard-imports it).
- Do NOT use `scripts/run_safe_pytest.sh` (wrong venv, wrong cwd — the
  sfpu_count_above header documents this).
- Do NOT pass `--speed-of-light` (the diag readback and goldens assume the
  runtime-parameter build; SOL also changes measured cycles).
- Do NOT pass `--enable-perf-counters` (mutually exclusive report kind; it
  overwrites the timing CSV).

## Step 0 — profiler sanity (device, run FIRST)

```bash
flock /tmp/tt-device.lock pytest --compile-producer -n 10 ./python_tests/test_profiler_overhead.py
flock /tmp/tt-device.lock pytest --compile-consumer ./python_tests/test_profiler_overhead.py
```

Marker pair must land at 30 ± 5 cycles on BH. If not, nothing below is
trustworthy.

## Step 1 — compile (no device; safe to run any time)

```bash
rm -f /tmp/cgtceq_bisect_rows.txt   # the bisect row dump appends across runs
pytest --compile-producer -n 10 -m perf ./python_tests/test_cgtceq_perf.py
```

Default sweep ≈ 48 variants (10 stream + 20 rendezvous/rate + ~18 bisect).
For the full ≥100-rows-per-distribution bisection statistics set
`CGTCEQ_BISECT_GROUPS=34` (adds ~90 variants) before BOTH phases.

## Step 2 — run (device; serialize under the device lock)

```bash
flock /tmp/tt-device.lock pytest --compile-consumer -m perf ./python_tests/test_cgtceq_perf.py
```

Deliberately NO `-n` on the consumer: the rendezvous/bisect tests read their
diagnostics back from device L1 (`read_from_device`) immediately after their
run; a parallel worker's run would overwrite the buffer between the two steps.
(The CI `-n 15` consumer pattern is fine for timing-only suites, not for this
one.)

Guardrail variant (recommended for the first run — new kernel, hang surface):

```bash
flock /tmp/tt-device.lock timeout 3600 pytest --compile-consumer -m perf \
    --timeout=300 --timeout-method=thread \
    ./python_tests/test_cgtceq_perf.py 2>&1 | tee /tmp/logs/cgtceq_consumer.log
```

All device-side polls in the kernel are BOUNDED (a dead ordering primitive or
a wrong Dst lane-map model reads out as a flagged assertion, not a hang), but
the MOP/replay machinery itself can still hang on a bad build — hence the
outer timeout. If a hang needs recovery: `~/flush.sh` (never while another
measurement is in flight).

## Step 3 — read the numbers (host-only)

```bash
python python_tests/cgtceq_analysis.py \
    --csv perf_data/test_cgtceq_perf/test_cgtceq_perf.csv \
    --rows /tmp/cgtceq_bisect_rows.txt
```

Validity gates, in order (sfpu_count_above discipline):

1. `ctrl_load` MATH_ISOLATE slope ≈ 1.0 cyc/vec (frontend floor). If not, the
   feed path is limiting and every other arm is uninterpretable.
2. `ctrl_swap` ≈ 2.0 cyc/vec (documented SFPSWAP bubble — the tripwire).
3. `rate` ≈ 2.0 cyc/vec (the CountD1 shape).
4. Only then read: the additivity deltas (expect single ≈ +2.0 on the ~3.94
   L1_TO_L1 floor, dual ≈ +4.0), the 3×3 rendezvous cycles/decision matrix
   (S0/R0 must reproduce ≥ 25.1 + fold + read; prior band 25–100), and the
   bisection p50/p95 (random ≈ 10–14 decisions + 1 cert).

Correctness is enforced inside the pytest run itself: the rendezvous arm's
per-segment counts are checked via an exact XOR-checksum automaton golden, and
every bisection row is checked field-by-field against an exact sign-magnitude
simulation (found threshold, Cgt, Ceq, decisions, exit mode, invariant
`Cgt < K <= Cgt+Ceq`). Any mismatch fails the test.

## Bring-up findings (2026-08-16 — suite is GREEN, 51/51)

The first consumer run (12 passed / 39 failed / 1 error) had TWO real bugs and
one schema gap; none of the three pre-declared levers (R0_WORD, diag flags,
SyncHalf base) was at fault — R0_WORD=0, the MMIO base, and the MMIO write
path (S2) are all confirmed correct by the passing self-checks.

1. **MATH_PACK semaphore leak (the hang).** Fill arms ended with
   `_llk_math_dest_section_done_`, which posts `semaphore::MATH_PACK`; the
   pack thread is idle in fill arms, so nothing consumed the token. The perf
   harness re-runs the kernel `run_count` times with no semaphore reset, and
   run 2's `_llk_math_pack_sync_init_` spins `while (semaphore_read(MATH_PACK)
   > 0)` forever → Math wedges in INIT, Unpacker wedges at the fill's
   `mailbox_read` ("waited 2 seconds for Math, Unpacker", Pack completes).
   The leaked token persists ACROSS TESTS on the un-reset device, which is why
   every test after the first rendezvous arm failed — including `rate-i2048`
   and all bisect cells, which were pure contamination. Fix: fill arms no
   longer call `_llk_math_dest_section_done_` (comment in the kernel).

2. **PerfConfig never writes stimuli (the wrong counts).** `TestConfig.run()`
   writes `variant_stimuli` to L1; `PerfConfig.run()` overrides run() and
   never does — the perf flow is timing-only by design. The self-checking
   arms were counting stale L1 (all counts read 2048 = "everything above a
   negative threshold", i.e. zeros/junk data). Fix: the driver calls
   `configuration.variant_stimuli.write(TestConfig.TENSIX_LOCATION)` itself
   before every device run (`_write_stimuli`). Any future perf test whose
   kernel READS its stimulus must do the same.

3. **CSV schema gates.** (a) run_types must be identical across every test in
   the module (rendezvous/bisect now run the same L1_TO_L1 + UNPACK_ISOLATE +
   MATH_ISOLATE triple as the stream tests; fill arms ignore PERF_RUN_TYPE so
   the extra run types are redundant-but-harmless re-measurements); (b)
   run_count must be uniform too — run_count=1 emits no `std(...)` columns
   and splits the schema (bisect now uses the shared `_RUN_COUNT`).

Diagnostics kept in the kernel (outside all timed zones): rendezvous INIT
captures MMIO words + twin-style per-tile SFPU counts into diag[9..15]
(printed by the driver as `[rdv] probe ...`); expected values with the
standard stimulus: cnt_t0_neg0=1024, cnt_t1_neg0=0, cnt_2t_neg0=1024,
cnt_t1_thr=1023.

**Measured constants (p150a, 2026-08-16):**

- Additivity (cyc/vec, tile_cnt slope): unpack floor L1_TO_L1(none)=4.079,
  UNPACK_ISOLATE=3.938 (the ~3.94 prior reproduces). single: MATH_ISOLATE
  2.438 vs L1_TO_L1 delta 2.420 → ADDITIVE; dual: 4.531 vs 4.515 → ADDITIVE.
  NOTE: the streamed controls read ctrl_load=1.406 / ctrl_swap=2.375 (not the
  1.0/2.0 loop-only floors) because the streamed shape pays a per-tile
  restart (sfpu_start + imm reloads + MOP re-issue) amortized over only 32
  vectors ≈ +0.4 cyc/vec; the pure-loop `rate` arm reads 2.0000 exactly,
  which is the CountD1 sanity gate actually satisfied.
- Rendezvous (cyc/decision, ITER slope ×64, exact-count-checked):

  | fold\sync | S0 tensix_sync | S1 sem+pcbuf | S2 sentinel |
  |---|---|---|---|
  | R0 full fold (read 1 word) | **81** | 101 | 98 |
  | R1 partial (read 16) | 132 | 157 | 151 |
  | R2 none (read 64) | 756 | 770 | 773 |

  The ≥25.1-cyc PassSync floor reproduces inside S0/R0's 81 (fold + 1 MMIO
  read + threshold-reload restart on top). MMIO Dst reads cost ~10 cyc/word
  (R1/R2 scaling), so fold-before-read is mandatory. The sentinel poll (S2)
  does NOT beat tensix_sync — its own MMIO polling burns the savings.
- Bisection (per row = 1 tile, K=32): random p50=14 decisions / 2313 cyc,
  p95=17 / 2960; clustered & allequal pin at 17 (16 probes + dual cert);
  specials p50=15. ≈165 cyc/decision ≈ count-pass (~64-70) + rendezvous
  (~81) + loop overhead. All 60 rows field-exact vs the sign-magnitude
  golden with the invariant Cgt < K <= Cgt+Ceq holding everywhere.

## Known bring-up levers (loud failures, one-line fixes)

- **`R0_WORD` (kernel #define, default 0):** which MMIO word of the scratch
  4-row window carries lane 0 after a full fold. If the SFPSTORE lane map
  differs from the model, the rendezvous checksum / bisect field checks fail
  (or flag 0x2/0x4 fires on the sentinel arm). Fix by sweeping `R0_WORD` over
  the even words 0..14 (add it to `CGTCEQ_PARAMS.convert_to_cpp`).
- **Flags word decode (diag[7]):** 0x1 semaphore-poll timeout, 0x2
  sentinel-poll timeout, 0x4 sentinel survived into the read. 0x2/0x4 on
  SYNC_PRIM=2 with S0/S1 passing ⇒ MMIO Dst *writes* don't land (the report
  verified reads verbatim; writes were the untested half) — drop the S2 column
  and note it.
- **Dest section base:** the MMIO model assumes SyncHalf section 0 = physical
  row 0 (the count_above/dprint precedent). If every count reads 0/garbage
  with flags==0, the section base is the suspect — add
  `DEST_REGISTER_HALF_SIZE` to the MMIO bases.

## NOTES — registration edits (none required)

- No conftest, CMake, or test-list edit is needed: pytest.ini already collects
  `test_*.py`, and the per-perf-test CSV schema gate
  (`test_perf_header_gate.py` → `perf_schema_derive.derive_perf_test_schemas`)
  only scans `perf_*.py` files, which this driver deliberately is not.
- IF the module is later renamed to `perf_cgtceq.py` (to appear in the CI perf
  shard), the gate will demand a catalog entry; apply this diff to
  `tests/python_tests/helpers/perf/test_schemas.py` (inside
  `PERF_TEST_SCHEMAS`, alphabetical position):

```python
    "perf_cgtceq": {
        "version": 1,
        "columns": [
            "arm", "dist", "dest_acc",
            "formats.input_A", "formats.input_B", "formats.output",
            "formats.register_A", "formats.register_B", "formats.sfpu_math",
            "fold", "iters", "k", "loop_factor", "marker", "num_faces",
            "num_faces_A", "num_faces_B", "relu_config", "rows", "seed",
            "sync", "thr2_bits", "thr_bits", "tile_cnt", "unpack_to_dest",
        ],
        "aliases": {},
    },
```

  (Column list must be re-derived at that point — run the gate and take its
  diff; the list above is the expected shape, not gospel.)

## Interactions with the Gate-1 sweep

Nothing in this bench was compiled or run while the Gate-1 baseline sweep was
in flight; no tracked file that JIT kernels depend on was edited. Producer
(step 1) is device-free; steps 0/2 must wait for the sweep to finish and take
`flock /tmp/tt-device.lock`.
