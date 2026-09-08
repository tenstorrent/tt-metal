# ttnop timing perturbation for Tensix kernels

`ttnop` looks for timing races in LLK kernels. Unpack, math, and pack run on
different Tensix threads; if synchronization is off by a cycle, the answer can
be wrong even when the code is fine. Those bugs often show up as flaky CI: the
test passes most runs and fails when timing shifts.

The tool patches one instruction in the loaded kernel, runs filler instructions
before that site resumes, and runs the test again. Change the site, filler type,
or delay count and you can usually make the failure repeat and narrow it down.
`report.md` records what moved and gives a command to rerun it.

This tree targets LLK Python tests. Exalens writes stimuli and reads results
straight into device L1, so the injector pokes detours there without rebuilding
the kernel for every delay count.

## Runners


| runner     | use it when                     | what it does                                                   |
| ---------- | ------------------------------- | -------------------------------------------------------------- |
| `ci.sh`    | many tests or a whole file      | compile once, try each site and delay once, shard across hosts |
| `focus.sh` | one test you already care about | same patch, many repeats, failure rate in the report           |


Same scanner, injector, and report format underneath.

Run from this directory:

```bash
./ci.sh --test test_eltwise_unary_datacopy.py --k Float16_b

./ci.sh --test test_sdpa_reinits.py \
    --splits 4 --group 2 \
    --report-dir reports/host2
```

For one pytest case and repeated runs, see [FOCUS.md](FOCUS.md).

`ci.sh` lives in this directory. It compiles the tests you point at, collects
pytest node ids, then sweeps every planned thread, site, filler, and delay.
Each case gets one clean run first. That output is the baseline. A supervisor
watches for hangs and wedges during breadth runs.

Usual path: run `ci.sh` on a file, read `report.md`, copy a **Reproduce** line
or hand the case to `focus.sh` if you need a rate.


| flag                   | default          | meaning                                             |
| ---------------------- | ---------------- | --------------------------------------------------- |
| `--test PATH`          | required         | Test file or directory. May be given more than once |
| `--k EXPR`             | no filter        | pytest `-k` filter                                  |
| `--markers EXPR`       | `PYTEST_MARKERS` | pytest marker filter                                |
| `--splits N --group G` | no split         | Shard G of N (pytest-split style)                   |
| `--jobs N`             | `15`             | compile workers                                     |
| `--device-jobs N`      | `8`              | pytest workers on device                            |
| `--report-dir DIR`     | `reports`        | where `failures.jsonl` and `report.md` go           |
| `--collect-to FILE`    | off              | compile and write node ids only, no device          |
| `--nodeids FILE`       | off              | sweep a saved id list (must already be compiled)    |


One `--report-dir` per machine when you split a sweep.

### `TTNOP_*` and `CHIP_ARCH`

The sweep reads its plan from environment variables. `sweep.py` builds the
`(thread, site, filler, delay)` grid from them on every pytest worker. The
shell scripts set or export these before calling pytest; you can also set them
yourself.

**What each variable controls**


| variable            | default                     | what it does                                                                                                                               |
| ------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `CHIP_ARCH`         | `wormhole`                  | Architecture for build and scanner binary (`wormhole`, `blackhole`, `quasar`). Quasar uses the simulator (`EXALENS_PORT`, default `5556`). |
| `TTNOP_SITE_MODE`   | `sync`                      | `sync` = sync/stall sites only. `all` = every safe instruction (slow).                                                                     |
| `TTNOP_THREADS`     | `unpack,math`               | Comma-separated thread ELFs to scan. Add `pack` if needed.                                                                                 |
| `TTNOP_SITES`       | empty = all                 | Limit to specific sites, e.g. `unpack:3,math:7`. Index is per thread; see [FOCUS.md](FOCUS.md).                                            |
| `TTNOP_DELAYS`      | `1-100`                     | Delay counts to try. Comma-separated values and inclusive ranges: `1-8,16,32`. Count `0` is added automatically when `TTNOP_REPEATS` > 1.  |
| `TTNOP_MAX_DELAY`   | `100`                       | Cave capacity in filler words. Delays above this are rejected; raise it only if the ELF has L1 room.                                       |
| `TTNOP_FILLER`      | `auto`                      | `auto`, `tti_nop`, `risc_nop`, `sfpnop`, `unpacr0`, `unpacr1`, `pacr`, or a raw hex word (`0x08000000`).                                   |
| `TTNOP_REPEATS`     | `1` (`10` in `focus.sh`)    | How many times to run each variant. This is the denominator in `failures / runs` in the report.                                            |
| `TTNOP_DRIFT`       | `1`                         | `1` = freeze stimuli and bit-compare to the clean run. `0` = let input change ([Drift](#drift)).                                           |
| `TTNOP_REPORT_DIR`  | `reports` / `reports/focus` | Where `failures.jsonl` and `report.md` go (`ci.sh` / `focus.sh`).                                                                          |
| `TTNOP_DEVICE_JOBS` | `8`                         | pytest workers during the device phase (one Tensix core each).                                                                             |
| `TTNOP_VERBOSE`     | off                         | Print every detour as it is armed.                                                                                                         |


`ci.sh` **only** (not read by the sweep planner itself):


| variable          | default              | what it does                                                         |
| ----------------- | -------------------- | -------------------------------------------------------------------- |
| `TTNOP_JOBS`      | `15`                 | CPU workers for `--compile-producer` (overridden by `ci.sh --jobs`). |
| `TTNOP_MARKERS`   | `PYTEST_MARKERS`     | pytest `-m` filter if `--markers` is not passed.                     |
| `TTNOP_STATE_DIR` | `$REPORT_DIR/.state` | Supervisor resume state; cleared when the run ends.                  |


**Examples**

Narrow a breadth run without editing the script:

```bash
./ci.sh --test test_mul_reduce_scalar.py \
    TTNOP_THREADS=math \
    TTNOP_FILLER=risc_nop \
    TTNOP_DELAYS=40-80
```

Same settings exported once, many invocations:

```bash
export TTNOP_THREADS=math TTNOP_FILLER=risc_nop TTNOP_DELAYS=40-80
./ci.sh --test test_mul_reduce_scalar.py
```

A `report.md` **Reproduce** block is just these variables plus `./focus.sh` on
one node id. Copy it as-is to rerun that site and delay band.

## What happens during a sweep

1. Compile selected tests (skip if `--nodeids` already has compiled cases).
2. Run each test clean once (baseline).
3. Scan each thread's ELF for sites and for the gap between `_etext` and
  `__loader_init_start` (scratch space in L1 for the cave).
4. For each variant, poke the detour into L1 through exalens: fillers in the
  cave, jump at the site. Changing delay is one word write, no reload.
5. Append findings to `failures.jsonl`.
6. Render `report.md`. `ci.sh` also writes `junit.xml` via the supervisor.

A case that fails clean is not swept. It needs an LLK ELF loaded. Cases that
call `run()` need a result to compare; cases that only call `run_elf_files()`
can assert in the test body.

## How the delay works

The scanner reads the ELF once. The tail of the L1 code region is the cave.
The linker reserves `[start, limit)` via `__kernel_cave_start` /
`__kernel_cave_end`, or the unused bytes after `_etext` when those symbols are
not present.

```text
cave start              -> filler instructions
displaced instruction   -> original instruction from the selected site
return                   -> jal x0, site + 4
```

Delay `n` starts `n` filler words before the displaced instruction. Only the
jump at the site changes when you change the count.

`ci.sh` runs each delay once (`TTNOP_REPEATS=1`). `focus.sh` defaults to 10
repeats per variant.

With repeats greater than 1, delay 0 is included too. It still jumps into the
cave but runs no fillers. If 0 passes and 8 fails, the fillers caused it, not
the detour alone.

Default cave size is 100 words (`TTNOP_MAX_DELAY`). Delays above that are
rejected up front, not clamped. Raise it only if the ELF has L1 room after
`_etext`.

`--device-jobs N` runs the device phase on N Tensix cores in parallel (N pytest
worker processes, each bound to one core).

## Filler instructions

A nop only widens a timing window if it retires on the unit whose timing is in
question. `tti_nop` advances the RISC and the front end together, `risc_nop` costs
the RISC a cycle while the backend keeps draining its instruction FIFO.


| filler     | word                                          | what it delays                |
| ---------- | --------------------------------------------- | ----------------------------- |
| `tti_nop`  | `0x08000000`                                  | RISC and the Tensix front end |
| `risc_nop` | `0x00000013`                                  | RISC only                     |
| `sfpnop`   | `0x3C000002`                                  | SFPU                          |
| `unpacr0`  | `0x0C000009` on WH/BH, `0x0C000005` on Quasar | unpacker 0 and SrcA           |
| `unpacr1`  | `0x0E000009` on WH/BH, `0x0C000405` on Quasar | unpacker 1 and SrcB           |
| `pacr`     | Quasar packer NOP via cfg RMW in the cave     | packer 0 on Quasar only       |


With `auto`, every thread tries `tti_nop` and `risc_nop`. Unpack sync sites also
try `unpacr0` and/or `unpacr1`. The scanner ORs the `CntSetMask` field of every
`SETADCXX` in `.text` to learn which unpackers actually read L1. An empty census
tries both unpacker nops rather than guessing. Math adds `sfpnop` at SFPU sites.
Pack adds `pacr` on Quasar only.

## Drift

Drift is how ttnop catches races that your test assertion misses.

A timing bug can move bits in the result buffer while the test still passes.
Many LLK tests check against a golden with tolerance or PCC, not bit equality.
The kernel can be wrong in a way the check does not see. Drift compares each
NOP run to the clean run on the same input and flags when those two tensors
differ even though pytest stayed green.

### Mismatch vs drift

A **mismatch** means the NOP run failed the test's own golden check. That check
is PCC checked against the reference golden.

A **drift** means the NOP run still passed that golden, but its hardware result
is not the same as the clean run's. Drift detection itself is bit-exact. Any
element that moved from clean to NOP counts, even if both tensors would still
pass a loose PCC gate against golden. The test's PCC vs
golden can be 0.99 while clean vs NOP might be much lower, and you would never
see it without drift.

When the tensors are numeric, the report also records **PCC vs clean**, between the
clean run and the NOP run on the same stimuli. That
number is not PCC vs golden. It tells you how far the race moved the output.
A Δ of `1 − pcc` near zero means a few bits flipped. A large Δ means the
buffers diverged badly even though the golden check still passed.

Drift findings go in `report.md` and `failures.jsonl` only. The pytest case
stays green and `ci.sh` still exits 0.

### Frozen stimuli

**RNG** (random number generator) is where `generate_stimuli()` and
`torch.rand` draw their input values from. With drift on, the plugin snapshots
PyTorch's default CPU RNG before the test body runs and rewinds to that snapshot
before every variant. All variants then replay the same random draws, so any
output change is timing, not new input data.

The clean run's device tensors are captured through a wrapped `TestConfig.run`
and compared bit for bit (NaN matches NaN).

`TTNOP_DRIFT=0` lets the RNG keep advancing between variants. That can still
find mismatches, but you can no longer compare NOP output to the clean run on the
same input.

## Reading the output

`failures.jsonl` is one JSON line per finding, appended as the run goes. Kill
the job and you still keep what finished.

`report.md` groups by test, site, and filler.

- delay counts that failed (often folded into ranges)
- failures / runs when repeats are on
- filler name and word
- finding type (`mismatch`, `drift`, `assert`, `error`, `hang`, `wedge`)
- DWARF call chain when available
- a `focus.sh` line to rerun that site

The report sorts by widest failing band first. One random failing count is less
useful than a run of counts.

`ci.sh` also emits `junit.xml`. The supervisor builds it if pytest dies
mid-run.

Only one process should write a report dir at a time. A new run renames
`failures.jsonl` to `failures.jsonl.prev` and drops old `report.md`,
`skips.jsonl`, and `junit.xml`. Use another `--report-dir` per branch or
experiment. `--collect-to` does not clear the report dir.

## Hangs and recovery

`ci.sh` runs a supervisor. A hang records the variant, then recovery tries a
spare core. If none is free it resets the card and resumes what was left.

Workers that go quiet inside a device call become `wedge` findings. A mismatch
can dirty state too, so the supervisor may move that worker before the next case.

Siblings of a hung param set are skipped (same race, same site). They are
listed in `skips.jsonl`, not as new findings.

## Exit status

`ci.sh`


| status | meaning                   |
| ------ | ------------------------- |
| `0`    | clean, or drift only      |
| `1`    | at least one real failure |
| `4`    | bad usage                 |
| `70`   | no `junit.xml`            |
| `75`   | wedge, rest finished      |
| `76`   | supervisor quit early     |




## Sanity check

If you changed the injector, scanner, or reporting, or you suspect they are
broken, run the bundled self-check from the `ttnop` directory.

```bash
make check
```

This runs `tests/ttnop_check.py` on silicon. It arms real L1 detours on a live
datacopy kernel and verifies each finding type the plugin would record.

- clean `tti_nop` detours on every scanned site restore without a finding
- a planted `ZEROSRC` filler is classified as `mismatch`
- a `tti_nop` detour plus a 1-bit source poke is classified as `drift`

```bash
make check-hang
```

The hang check passes `--hang` to the same script. A self-jumping cave filler
wedges a core and should be classified as `hang`. Run this only when you can
reset the card afterward. It runs `tt-smi -r` when it finishes.
