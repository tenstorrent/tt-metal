# ttnop timing perturbation for Tensix kernels

`ttnop` helps find timing races in LLK kernels. It adds a small delay at one
instruction, runs the test again, and records which delays change the result.
This turns an occasional CI failure into a repeatable test case that is easier
to debug.

The tool runs LLK Python tests.

## Choose a runner

There are two ways to run the tool.

| runner | use it when | what it does |
| --- | --- | --- |
| `ci.sh` | You want to search many tests | Runs each planned delay once and can split work across hosts |
| `focus.sh` | You already have one failing test or report entry | Repeats selected delays and shows a failure rate |

Both runners compile first and take a device lock. They write a report when
they find something. `ci.sh` also uses a supervisor so it can recover when a
Tensix core hangs.

Run these commands from this directory.

```bash
# Search one test file
./ci.sh --test test_eltwise_unary_datacopy.py --k Float16_b

# Search one part of a suite
./ci.sh --test test_sdpa_reinits.py \
    --splits 4 --group 2 \
    --report-dir reports/host2

# Measure one site and two delay counts
./focus.sh --sites unpack:3 --nop risc_nop --delays 8,16 \
    'test_x.py::test_y[params]'
```

Always quote a pytest node id. Parameterized ids often contain brackets and
other characters that the shell treats specially.

See [FOCUS.md](FOCUS.md) for the full `focus.sh` guide.

## Breadth runner options

`ci.sh` accepts one or more `--test` paths. Use `--test .` to select the full
LLK Python test suite.

| flag | default | meaning |
| --- | --- | --- |
| `--test PATH` | required | Test file or directory. May be given more than once |
| `--k EXPR` | no filter | pytest name filter |
| `--markers EXPR` | `PYTEST_MARKERS` | pytest marker filter |
| `--splits N --group G` | no split | Select group G from N pytest-split groups |
| `--jobs N` | `15` | CPU workers used for compilation |
| `--device-jobs N` | `8` | xdist workers used for the sweep |
| `--report-dir DIR` | `reports` | Output directory for this run |
| `--collect-to FILE` | disabled | Compile and save node ids without using the device |
| `--nodeids FILE` | disabled | Sweep a saved node-id list. The tests must already be compiled |

Give each host its own report directory when a sweep is split across machines.
`ci.sh` also accepts settings such as `TTNOP_DELAYS=1-20` and
`CHIP_ARCH=blackhole` as command-line arguments.

## What happens during a sweep

1. The runner compiles the selected tests.
2. Each test runs once without a detour. This is the clean baseline.
3. The scanner finds candidate instructions in the selected thread ELFs.
4. The plugin points a site at the code cave and runs each planned delay.
5. Failures and output changes are appended to `failures.jsonl`.
6. The records are turned into `report.md`.

Tests that fail during the clean baseline are not swept. A test must load an
LLK ELF. Tests that call `run()` also need a result to compare. Tests that call
`run_elf_files()` directly can use assertions in their test body.

## How the delay works

The ELF is scanned once. The unused tail of its L1 code region is used as a
small code cave.

```text
cave start              -> filler instructions
displaced instruction   -> original instruction from the selected site
return                   -> jal x0, site + 4
```

The selected instruction is replaced with a jump into the cave. A delay of
`n` starts `n` filler words before the displaced instruction. Changing the
delay only changes that jump, so the kernel does not need to be rebuilt or
reloaded for every count.

Any run with more than one repeat also includes delay 0. Delay 0 still takes
the jump but skips every filler. If delay 0 passes and delay 8 fails, the filler
caused the change rather than the detour itself.

The default cave holds 100 filler words. Raise `TTNOP_MAX_DELAY` if you need a
larger delay, but the ELF must have enough free L1 space.

## Filler instructions

Different fillers delay different parts of Tensix.

| filler | word | what it delays |
| --- | --- | --- |
| `tti_nop` | `0x08000000` | RISC and the Tensix front end |
| `risc_nop` | `0x00000013` | RISC only |
| `sfpnop` | `0x3C000002` | SFPU |
| `unpacr0` | `0x0C000009` on WH/BH, `0x0C000005` on Quasar | unpacker 0 and SrcA |
| `unpacr1` | `0x0E000009` on WH/BH, `0x0C000405` on Quasar | unpacker 1 and SrcB |

The `auto` policy always tries `tti_nop` and `risc_nop`. It also adds the
active unpacker filler when scanning unpack sync sites. It adds `sfpnop` at
SFPU sites on architectures where the scanner identifies them. Pack uses only
`tti_nop` and `risc_nop`. If the scanner cannot tell which unpacker is active,
it tries both.

Only pure unpacker NOP mode is used. Modes such as `ZEROSRC`, `SET_DVALID`, and
`NEGINFSRC` change source state and would test data corruption instead of
timing.

`risc_nop` may need a wider delay range because the instruction FIFO can absorb
the first few cycles. Do not assume a site is safe after checking only a few
small counts.

## Main settings

The runners read these environment variables. `focus.sh` has matching flags for
the settings used most often.

| variable | default | meaning |
| --- | --- | --- |
| `CHIP_ARCH` | `wormhole` | Architecture used for the build and device lock |
| `TTNOP_SITE_MODE` | `sync` | Use `sync` for sync and stall sites, or `all` for every safe candidate |
| `TTNOP_THREADS` | `unpack,math` | Thread ELFs to scan. Add `pack` when needed |
| `TTNOP_SITES` | all found sites | Specific indices such as `unpack:3,math:7` |
| `TTNOP_DELAYS` | `1-100` | Counts and ranges such as `1-8,16,32` |
| `TTNOP_MAX_DELAY` | `100` | Number of filler words reserved in the cave |
| `TTNOP_FILLER` | `auto` | A filler name or raw instruction word |
| `TTNOP_REPEATS` | `1` | Runs per variant. `focus.sh` changes this default to `10` |
| `TTNOP_DRIFT` | `1` | Freeze input data and compare output with the clean run |
| `TTNOP_REPORT_DIR` | runner-specific | `reports` for `ci.sh` and `reports/focus` for `focus.sh` |
| `TTNOP_DEVICE_JOBS` | `8` | Tensix cores used during the device phase |
| `TTNOP_VERBOSE` | off | Print every detour as it is armed |
| `TTNOP_BUILD_LOCK` | `/tmp/ttnop-build-$CHIP_ARCH.lock` | Lock shared by compile producers |
| `TTNOP_DEVICE_LOCK` | `/tmp/tt-llk-test-$CHIP_ARCH.lock` | Lock that keeps sweeps off the same card |

`TTNOP_DELAYS` accepts comma-separated counts and inclusive ranges. For example,
`1-8,16,32` runs counts 1 through 8, then 16 and 32.
Counts above `TTNOP_MAX_DELAY` are rejected instead of being shortened.

Use every count when possible. Timing races often appear in a band, and a short
list of powers of two can miss that band.

With `CHIP_ARCH=quasar`, the runners add `--run-simulator` and use
`EXALENS_PORT`, which defaults to `5556`. Wormhole and Blackhole use silicon.

## Drift

A test may still pass its tolerance check even when the output bits changed.
With drift checking enabled, every variant uses the same random input as the
clean baseline and is compared with that baseline bit for bit.

A drift record means the output changed but the test still passed its own
golden check. Drift is written to the report and does not make the pytest case
fail.

Some tests cannot reproduce the same bits by design. The plugin disables drift
for cases the LLK harness marks as unsupported. It also runs one extra clean
control before trusting drift results. If that control changes, drift is
disabled for the case.

Set `TTNOP_DRIFT=0` or pass `focus.sh --no-drift` to let the random stream move
between variants. This can explore more input values, but the outputs can no
longer be compared with the baseline.

## Reading the output

`failures.jsonl` is the source data. Each line is written as soon as a variant
is recorded, so completed findings survive a stopped run.

`report.md` is only written when the run records a finding or a supervised skip.
It groups those records by test, site, and filler. It includes:

- the widest band of failing delay counts
- the highest failure rate when repeats are enabled
- the failure type, such as `mismatch`, `drift`, `assert`, `error`, `hang`, or `wedge`
- the source call chain from DWARF when it is available
- a `focus.sh` command for reproducing each finding

The widest band is usually the best reproducer. A single failing count may only
hit the bad timing by chance.

Supervised `ci.sh` runs also write `junit.xml`. The supervisor builds it from
per-case records because a killed pytest session cannot write a complete JUnit
file on its own.

Each run locks its report directory. This prevents two branches from appending
to the same files at the same time. Before a new run, `failures.jsonl` is moved
to `failures.jsonl.prev`. Old `report.md`, `skips.jsonl`, and `junit.xml` files
are removed. Only one previous failure log is kept. `ci.sh --collect-to` does
not reset the report directory. Use a different `--report-dir` when you need
to keep more runs.

## Hangs and recovery

`ci.sh` watches every pytest worker. When a variant hangs a core, the worker
records the variant and asks the supervisor for recovery. The supervisor first
tries to remove that worker so xdist can replace it on a spare core. If that is
not possible, it stops the attempt, resets the card, and resumes the unfinished
cases.

The supervisor also catches workers that stop responding inside a device call.
These show up as `wedge` records. A mismatch can leave device state dirty, so a
supervised run also moves that worker to a spare core before giving it another
case.

Cases skipped because a sibling already hung the same test family are saved in
`skips.jsonl` and listed in the report. They are not separate findings.

`focus.sh` has no supervisor. It records the hang and ends the case, but it
cannot free the hung core. Reset the card before using it again.

`focus.sh` normally uses pytest exit statuses. The additional supervisor
statuses below only come from `ci.sh`.

| status | meaning |
| --- | --- |
| `0` | Clean run, possibly with drift-only records |
| `1` | At least one non-drift variant failed |
| `4` | Invalid command line |
| `70` | `ci.sh` did not get a JUnit file |
| `75` | A wedge was recorded and the remaining cases finished |
| `76` | The supervisor stopped with unfinished cases |

## Checking changes to the tool

Run the silicon check after changing scanner or injector behavior.

```bash
make check
```

This builds one datacopy kernel and checks clean detours, forced mismatches, and
drift detection on a card.

The hang check intentionally wedges a core. It tries to reset the card when it
finishes and tells you if a manual reset is still needed.

```bash
make check-hang
```

## Current limit

Metal kernels are not supported. Metal copies its cached host binaries into L1
on every slow-dispatch launch, which overwrites a device-side patch before the
kernel runs. Supporting Metal needs a host-side patch point between kernel
configuration and the go signal.
