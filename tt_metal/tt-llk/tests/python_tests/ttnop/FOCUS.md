# Using focus.sh on one test case

Use `focus.sh` after a broad sweep finds a timing-sensitive test. It runs one
pytest case many times with selected delays, which shows how often the problem
happens instead of giving only one pass or fail result.

For example, 3 failures from 10 repeats gives a failure rate of 3/10. This tool
only supports LLK Python tests. It cannot run Metal tests.

The easiest starting point is the `Reproduce` command in `report.md`. It already
contains the test, thread, site, filler, and delay range for one finding.

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
`file.py::test_name[parameters]`. The quotes matter because brackets and other
characters have special meaning to the shell.

## What the command does

1. Compiles the selected test in the shared LLK build tree.
2. Takes the device lock so another sweep cannot use the card.
3. Runs the test once without a detour to get a clean baseline.
4. Scans the ELF binary for each selected Tensix thread, such as unpack or math.
5. Builds a plan from the selected sites, fillers, delays, and repeat count.
6. Splits that plan across the requested Tensix cores.
7. Records findings in `failures.jsonl` and writes `report.md`.

The clean run must pass. If it fails before a detour is added, there is no timing
result to measure and the case stays failed.

Compilation uses a shared build lock because the compile producer updates the
shared LLK build tree. A run may wait here if another producer is active.

## Narrow the run

A default focused run can be large. It scans every sync site in the unpack and
math threads, tries the automatic filler set, sweeps delays 1 through 100, and
runs each variant 10 times.

Start by choosing a site, a filler, and a useful delay range.

| flag | environment variable | default | meaning |
| --- | --- | --- | --- |
| `--thread`, `--threads` | `TTNOP_THREADS` | `unpack,math` | Thread ELFs to scan. Use a comma-separated list |
| `--site`, `--site-mode` | `TTNOP_SITE_MODE` | `sync` | Use `sync` for sync and stall sites, or `all` for every safe candidate |
| `--sites` | `TTNOP_SITES` | all found sites | Site indices such as `unpack:3,math:7` |
| `--nop`, `--filler` | `TTNOP_FILLER` | `auto` | One filler type, a raw instruction word, or `auto` |
| `--delays` | `TTNOP_DELAYS` | `1-100` | Counts and inclusive ranges such as `1-8,16,32` |

To keep unpack and math while adding pack, use
`--threads unpack,math,pack`. Using `--threads pack` selects only pack. BRISC
and NCRISC reader or writer kernels are not supported.

Site indices start at 0 and depend on the site mode. With `--site sync`,
`unpack:3` selects the fourth sync or stall site in the unpack ELF. With
`--site all`, it selects the fourth safe candidate instruction.

The `auto` filler policy tries `tti_nop` and `risc_nop`. It also tries the active
unpacker NOP at unpack sync sites and `sfpnop` at SFPU sites when the scanner
can identify them. Pack uses only `tti_nop` and `risc_nop`.

Naming one filler is the quickest way to reduce the run. Different fillers
delay different hardware units. A race may react strongly to one filler and
not another.

Use `--site all` carefully. A kernel may have hundreds of candidate
instructions.

## Repeats and delay 0

`focus.sh` uses 10 repeats by default. The failure rate in the report is the
number of failed repeats divided by this value.

When repeats are greater than 1, the plan adds delay 0 for each site and filler.
Delay 0 still takes the jump through the cave but runs no filler instructions.
This checks whether the detour itself changes the test.

If delay 0 passes and delay 54 fails, the filler delay is the useful signal.

## Other options

| flag | environment variable | default | meaning |
| --- | --- | --- | --- |
| `--repeats` | `TTNOP_REPEATS` | `10` | Runs per variant |
| `--device-jobs` | `TTNOP_DEVICE_JOBS` | `8` | xdist workers, each assigned to one Tensix core |
| `--max-delay` | `TTNOP_MAX_DELAY` | `100` | Filler capacity of the code cave |
| `--no-drift` | `TTNOP_DRIFT` | enabled | Use different random input for each variant and stop checking output drift |
| `--report-dir` | `TTNOP_REPORT_DIR` | `reports/focus` | Output directory |
| `--verbose` | `TTNOP_VERBOSE` | disabled | Print every detour before it runs |

`CHIP_ARCH` defaults to `wormhole` and selects the scanner, build, and device
lock. `CHIP_ARCH=quasar` runs through the simulator on `EXALENS_PORT`, which
defaults to `5556`.

Delays above `TTNOP_MAX_DELAY` are rejected. They are not reduced to fit.

Flags and environment variables can be mixed. A flag replaces an inherited
value for the same setting.

```bash
TTNOP_SITES=unpack:3 TTNOP_DELAYS=8,16 ./focus.sh \
    'test_x.py::test_y[params]'
```

## Estimate the work

The rough number of test body runs is:

```text
sites x fillers x delay counts x repeats
```

Delay 0 adds one more delay count when repeats are greater than 1.

For one unpack site with three automatic fillers and 10 repeats:

| selection | test body runs | about each core with 8 workers |
| --- | --- | --- |
| `--delays 54` | 20 per filler, 60 total | 8 |
| `--nop risc_nop --delays 40-60` | 220 | 28 |
| `--delays 1-100` | 3030 | 379 |

Use the numbers as an estimate. Sites can have a different number of automatic
fillers. If the scanner cannot identify the active unpacker, an unpack sync site
gets four filler types instead of three.

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
band width, it prefers the higher failure rate. Start with the first reproduce
command under the site.

Finding types include:

| type | meaning |
| --- | --- |
| `mismatch` | The test failed its normal golden check |
| `drift` | The test passed its golden check, but output differed from the clean run |
| `assert` | The LLK harness raised an assertion |
| `error` | The variant raised another exception |
| `hang` | The device call timed out and the core may be stuck |
| `wedge` | A `ci.sh` supervisor found a worker that stopped responding |

Drift-only findings do not fail the pytest case.

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

`focus.sh` does not run the supervisor. With the default 8 workers, a hung
worker stops its part of the plan while the other workers continue on their
own cores. The script cannot replace the hung worker or reset the card.

Reset the card before running more device tests after a hang:

```bash
tt-smi -r
```

Use `ci.sh` when a large sweep needs automatic worker replacement, card reset,
and resume support.

## Exit status

- `0` means the case stayed clean or only produced drift records
- `1` means at least one non-drift variant failed
- `4` means the command line was invalid

Other pytest errors can also produce a nonzero status. Read the terminal output
before treating a failed command as a timing finding.
