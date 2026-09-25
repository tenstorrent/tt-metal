# Using focus.sh on one test case

`focus.sh` perturbs one pytest case in depth to expose timing races. It varies
injection sites, filler instructions, and delay counts, then runs each variant
many times. That repeated perturbation surfaces non-deterministic failures and
hangs that a single run might miss.

The report shows how often each variant failed or hung. For example, 3 failures
from 10 repeats is a failure rate of 3/10.

This tool only supports LLK Python tests. It cannot run Metal tests.

## Quick start

Run from the `ttnop` directory and quote the full pytest node id.

```bash
./focus.sh \
    'test_eltwise_unary_datacopy.py::test_unary_datacopy[formats:Float16_b->Float16_b-dest_acc:No-num_faces:1-tilize:No-input_dimensions:[64, 64]]'
```

To measure a known site with one filler and a small delay range:

```bash
./focus.sh \
    --sites unpack:3 \
    --nop risc_nop \
    --delays 40-60 \
    'test_mul_reduce_scalar.py::test_mul_reduce_scalar[formats:Float16_b->Float16_b-math_fidelity:HiFi2-num_tiles:1-tile_dimensions:[16, 16]]'
```

The pytest node id is the full test name in the command, starting with the test
file and ending with any bracketed parameters. It has the form
`file.py::test_name[parameters]`. Quote it: brackets and other characters are
special to the shell.

## What the command does

1. Compiles the selected test in the shared LLK build tree.
2. Runs the test once without a detour to get a clean baseline.
3. Scans each selected thread's ELF for detour sites and for the gap between
  the `_etext` and `__loader_init_start` linker markers. For LLK tests, the
   space between these markers is the unused tail of the kernel's fixed L1 code
   region.
4. Builds a plan from the selected sites, fillers, delays, and repeat count, and
  splits it across the requested Tensix cores.
5. Pokes each variant directly into device L1 through exalens: filler instructions
  go into the cave region, and one site instruction is replaced with a jump
   into that cave. The harness keeps the same ELF image in L1 across reruns, so
   changing the delay is one word write with no rebuild or reload.
6. Records findings in `failures.jsonl`, including how often each variant
  failed or hung, and writes `report.md`.

The clean run must pass. If it fails before a detour is added, there is no timing
result to measure and the case stays failed.

## Narrow the run

A default focused run can be large. It scans every sync site in the unpack and
math threads, tries the automatic filler set, tests delays 1 through 100, and
runs each variant 10 times.

Start by choosing a site, a filler, and a useful delay range.

Every setting below can be passed either as a `focus.sh` flag or as a
`TTNOP_*` environment variable. Use whichever is more convenient.

```bash
TTNOP_SITES=unpack:3 TTNOP_DELAYS=8,16 ./focus.sh \
    --nop risc_nop \
    'test_x.py::test_y[params]'
```

See [README.md](README.md) for shared concepts such as drift, filler words, and
cave layout.


| flag                    | environment variable | default         | meaning                                                                |
| ----------------------- | -------------------- | --------------- | ---------------------------------------------------------------------- |
| `--thread`, `--threads` | `TTNOP_THREADS`      | `unpack,math`   | Thread ELFs to scan. Use a comma-separated list                        |
| `--site`, `--site-mode` | `TTNOP_SITE_MODE`    | `sync`          | Use `sync` for sync and stall sites, or `all` for every safe candidate |
| `--sites`               | `TTNOP_SITES`        | all found sites | Site indices such as `unpack:3,math:7`                                 |
| `--nop`, `--filler`     | `TTNOP_FILLER`       | `auto`          | One filler type, a raw instruction word, or `auto`                     |
| `--enable-unpacr-nop`   | `TTNOP_ENABLE_UNPACR_NOP` | disabled  | Add UNPACR0/1 NOPs to unpack sync sites for an explicit hardware audit |
| `--delays`              | `TTNOP_DELAYS`       | `1-100`         | Counts and inclusive ranges such as `1-8,16,32`                        |


To keep unpack and math while adding pack, use
`--threads unpack,math,pack`. Using `--threads pack` selects only pack. BRISC
and NCRISC reader or writer kernels are not supported for LLK tests because
exalens writes stimuli and reads results directly in L1 (no data-movement
kernels are involved).

Site indices start at 0 and depend on the site mode. With `--site sync`,
`unpack:3` selects the fourth sync or stall site in the unpack ELF. With
`--site all`, it selects the fourth safe instruction.

With `auto`, every thread always tries `tti_nop` and `risc_nop`. Math sites add
`sfpnop` at SFPU instructions when the scanner can identify them on Wormhole
and Blackhole. Pack adds `pacr` on Quasar only.

UNPACR0/1 NOPs are deliberately excluded from normal runs. They execute through
the full unpacker pipeline and may wait for a free source bank, so findings have
a history of looking like hardware issues when they are actually software
behavior. Use `--enable-unpacr-nop` only for an explicit hardware audit.

Use `--site all` carefully. A kernel may have hundreds of candidate
instructions, which will take a very long time to complete.

## Other options


| flag            | environment variable | default         | meaning                                                            |
| --------------- | -------------------- | --------------- | ------------------------------------------------------------------ |
| `--repeats`     | `TTNOP_REPEATS`      | `10`            | Runs per variant                                                   |
| `--device-jobs` | `TTNOP_DEVICE_JOBS`  | `8`             | Pytest workers, each assigned to one Tensix core                   |
| `--max-delay`   | `TTNOP_MAX_DELAY`    | `100`           | Filler capacity of the code cave                                   |
| `--no-drift`    | `TTNOP_DRIFT`        | enabled         | Turn off drift checking                                            |
| `--report-dir`  | `TTNOP_REPORT_DIR`   | `reports/focus` | Output directory                                                   |
| `--verbose`     | `TTNOP_VERBOSE`      | disabled        | Print every detour before it runs                                  |




## Estimate the work

The rough number of test body runs is:

```text
sites x fillers x delay counts x repeats
```

With more than one repeat, delay 0 is added as a control: the jump still runs,
but no fillers do. The totals below include it and exclude clean baseline and
reproducibility runs. Repeats for a variant stay on one core.

For one unpack site with the two default automatic fillers and 10 repeats:


| selection                       | test body runs          |
| ------------------------------- | ----------------------- |
| `--delays 54`                   | 20 per filler, 40 total |
| `--nop risc_nop --delays 40-60` | 220                     |
| `--delays 1-100`                | 2020                    |




## Reading the report

`failures.jsonl` contains one record for every variant that produced a finding.
The file is updated during the run, so completed findings remain if the command
is stopped.

`report.md` is the readable view. For each site it shows:

- delay counts that produced a finding
- failures divided by runs
- the filler instruction and exact word
- the finding type
- the source call chain when DWARF information is available
- a command that repeats only that finding

The report puts the widest failing band first. If two fillers have the same
band width, it prefers the higher failure rate.

Finding types include:


| type       | meaning                                                                                               |
| ---------- | ----------------------------------------------------------------------------------------------------- |
| `mismatch` | A Python assertion or `pytest.fail()` failed the test                                                 |
| `drift`    | Output differed from the clean run but the golden still passed                                        |
| `assert`   | The LLK harness raised an assertion                                                                   |
| `error`    | The variant raised another exception                                                                  |
| `hang`     | The device call timed out and the core may be stuck                                                   |
| `wedge`    | A worker stopped responding and may need a device reset                                               |


Drift-only findings do not fail the pytest case.

### Example

Below is a shortened excerpt from a real `report.md`.

**Strongest signal**

```text
widest band: unpack STALLWAIT@0x067d8 unpacr0 13, 18-19, 27-28, …, 96-98 (41/41)
```

**Sites** (summarized — one test case shown)


| #   | thread | site                | NOP_TYPE   | NOP counts               | how      |
| --- | ------ | ------------------- | ---------- | ------------------------ | -------- |
| 1a  | unpack | `STALLWAIT@0x067d8` | `unpacr0`  | 13, 18-19, …, 96-98 (41) | mismatch |
| 1b  | unpack | `SEMGET@0x067f0`    | `unpacr0`  | 13, 18 (2)               | mismatch |
| 1c  | math   | `STALLWAIT@0x0b2c4` | `tti_nop`  | 39-41, 43, …, 96-98 (32) | mismatch |
| 1c  | math   | `STALLWAIT@0x0b2c4` | `risc_nop` | 86, 90-91, 93, 96-98 (7) | mismatch |


Each site then gets its own section. Site `1a` from the table above expands to:

**1a. unpack STALLWAIT @ 0x067d8**

- case: `test_hadamard.py::test_hadamard_h128[fidelity:LoFi-normalize:False-num_tiles:8]`
- site index (mode `sync`): 2
- first finding: `H128 result is not exactly H_128 @ x`

NOP types:


| NOP_TYPE  | word         | NOP counts                      | how      |
| --------- | ------------ | ------------------------------- | -------- |
| `unpacr0` | `0x0c000009` | 13, 18-19, 27-28, …, 96-98 (41) | mismatch |


Where the NOPs went in (innermost frame first):

1. `_llk_unpack_hadamard_h128_  tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_unpack_hadamard.h:125`
2. `run_kernel  tt_metal/tt-llk/tests/sources/hadamard_test.cpp:42`

Reproduce:

```bash
CHIP_ARCH=blackhole TTNOP_SITE_MODE=sync TTNOP_THREADS=unpack TTNOP_SITES=unpack:2 \
  TTNOP_ENABLE_UNPACR_NOP=1 TTNOP_DELAYS=13,18-19,27-28,33-34,36-37,39-41,43,46-47,50,52,54,56-57,59,61-63,65-66,69,72-73,75-76,78,80,83,86,90-91,93,96-98 \
  ./focus.sh 'test_hadamard.py::test_hadamard_h128[fidelity:LoFi-normalize:False-num_tiles:8]'
```



## Reports from separate runs

Only one process can use a report directory at a time. This prevents two
branches or two terminals from appending to the same report.

At the start of a run, the previous `failures.jsonl` is renamed to
`failures.jsonl.prev`. Old `report.md`, `skips.jsonl`, and `junit.xml` files are
removed.

Choose another directory when you want to keep several runs.

```bash
./focus.sh --report-dir reports/issue-123 \
    --sites unpack:3 --nop risc_nop --delays 40-60 \
    'test_x.py::test_y[params]'
```



## Hangs

With the default 8 workers, a hung
worker stops its part of the plan while the other workers continue on their
own cores. The script cannot replace the hung worker or reset the card.

Reset the card before running more device tests after a hang:

```bash
tt-smi -r
```



## Sanity check

If you suspect the injector, scanner, or reporting is broken, run the bundled
self-check from the `ttnop` directory:

```bash
make check
```

This runs `tests/ttnop_check.py`, which arms real L1 detours on a live datacopy
kernel and checks restoration and selected finding types:

- clean `tti_nop` detours on every scanned site restore without a finding
- a planted `ZEROSRC` filler is classified as `mismatch`
- a `tti_nop` detour plus a 1-bit source poke is classified as `drift`
- with `--hang`, a self-jumping cave filler is classified as `hang` (wedges a
core — run only when you can reset the card afterward)

```bash
make check-hang
```

The hang check runs `tt-smi -r` when it finishes. If a previous run left the
core wedged, reset the card before `make check`.

## Exit status

- `0` means the case stayed clean or only produced drift records
- `1` means the clean baseline or a non-drift variant failed

Other pytest errors can also produce a nonzero status. Read the terminal output
before treating a failed command as a timing finding.
