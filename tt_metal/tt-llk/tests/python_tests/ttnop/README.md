# ttnop timing perturbation for Tensix kernels

`ttnop` looks for timing races in LLK kernels. Unpack, math, and pack run on
different Tensix threads; if synchronization is off by a cycle, the answer can
be wrong even when ordinary runs pass. Use `focus.sh` when you suspect a race
and want to see whether changing the timing exposes it.

The tool patches one instruction in the loaded kernel, runs filler instructions
before that site resumes, and runs the test again. Change the site, filler type,
or delay count and you can usually make the failure repeat and narrow it down.
`report.md` records what moved and gives a command to rerun it.

This tree targets LLK Python tests. Exalens writes stimuli and reads results
straight into device L1, so the injector pokes detours there without rebuilding
the kernel for every delay count.

Run `focus.sh` on one pytest case, read `report.md`, then copy a **Reproduce**
command to narrow down a finding. See [FOCUS.md](FOCUS.md) for commands and options.

`ci.sh` also exists for sweeping many tests in CI, using the same scanner,
injector, and report format.

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
| `TTNOP_SITES`       | empty = all                 | Limit to specific sites, e.g. `unpack:3,math:7`. Index is per thread                                            |
| `TTNOP_DELAYS`      | `1-100`                     | Delay counts to try. Comma-separated values and inclusive ranges: `1-8,16,32`.  |
| `TTNOP_MAX_DELAY`   | `100`                       | Cave capacity in filler words. Delays above this are rejected; raise it only if the ELF has L1 room.                                       |
| `TTNOP_FILLER`      | `auto`                      | `auto`, `tti_nop`, `risc_nop`, `sfpnop`, `pacr`, or a raw hex word (`0x08000000`). UNPACR NOPs cannot be selected here.                     |
| `TTNOP_ENABLE_UNPACR_NOP` | off                   | `1` adds UNPACR0/1 NOPs to automatic unpack-thread sync-site sweeps. Intended for explicit hardware audits only.                           |
| `TTNOP_REPEATS`     | `1` (`10` in `focus.sh`)    | How many times to run each variant. This is the denominator in `failures / runs` in the report.                                            |
| `TTNOP_DRIFT`       | `1`                         | `1` = replay the CPU RNG and compare values to the clean run.                                    |
| `TTNOP_REPORT_DIR`  | `reports` / `reports/focus` | Where `failures.jsonl` and `report.md` go (`ci.sh` / `focus.sh`).                                                                          |
| `TTNOP_DEVICE_JOBS` | `8`                         | pytest workers during the device phase (one Tensix core each).                                                                             |
| `TTNOP_VERBOSE`     | off                         | Print every detour as it is armed.                                                                                                         |


A `report.md` **Reproduce** block is just these variables plus `./focus.sh` on
one node id. Copy it as-is to rerun that site and delay band.

A pytest node id uniquely identifies one collected test, including its parameter
values, for example `test_file.py::test_name[param]`.

## What happens during a sweep

1. Compile the selected test.
2. Run each test clean once (baseline).
3. Scan each thread's ELF for sites and for the gap between `_etext` and
  `__loader_init_start` (scratch space in L1 for the cave).
4. For each variant, poke the detour into L1 through exalens: fillers in the
  cave, jump at the site. Changing delay is one word write, no reload.
5. Append findings to `failures.jsonl`.
6. Render `report.md`.

A case that fails clean is not swept. It needs an LLK ELF loaded. Cases that
call `run()` need a result to compare; cases that only call `run_elf_files()`
can assert in the test body.

## How the delay works

The scanner reads the ELF once. The tail of the L1 code region is the cave.
The scanner uses the gap from `_etext` (aligned to 16 bytes) to
`__loader_init_start`. The injector checks that the detour fits in that space.

```text
cave start              -> 100-word filler area
clear                   -> another filler, or Quasar PACR config clear
displaced instruction   -> instruction moved from the selected site
return                  -> jal x0, site + 4
Quasar PACR setup       -> config set; jal x0, first PACR filler
```

Delay `n` starts `n` filler words before the displaced instruction. Only the
jump at the site changes when you change the count.

The Quasar PACR setup is physically last in the cave but executes first because
the patched site jumps directly to it. It sets packer 0 to no-write mode, jumps
backward through exactly `n` PACR_STRIDEs, clears the config in `clear`
slot, executes the displaced instruction, then jumps back to the original
kernel. Delay 0 bypasses the PACR setup and filler run.

`focus.sh` defaults to 10 repeats per variant.

With repeats greater than 1, delay 0 is included too. It still jumps into the
cave but runs no fillers. If 0 passes and 8 fails repeatedly, that suggests
the added delay exposes timing sensitivity; it does not prove the root cause.

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
| `pacr`     | Quasar PACR_STRIDE bracketed by config RMW    | packer 0 on Quasar only       |


With `auto`, every thread tries `tti_nop` and `risc_nop`. Math adds `sfpnop` at
SFPU sites. Pack adds `pacr` on Quasar.

UNPACR0/1 NOPs are off by default because they can expose software behavior
that looks like a hardware bug. Unlike a simple one-cycle NOP, they travel
through the full unpacker pipeline and may wait for a free source bank.Enable
them only for hardware audits with `--enable-unpacr-nop` or `TTNOP_ENABLE_UNPACR_NOP=1`.

When enabled, the scanner uses the `CntSetMask` fields in `SETADCXX` instructions
to identify which unpackers read L1. If it cannot identify one, it tries both.

## Drift

Drift is how ttnop catches races that your test assertion misses.

A timing bug can move bits in the result buffer while the test still passes.
Many LLK tests check against a golden with tolerance or PCC, not bit equality.
The kernel can be wrong in a way the check does not see. Drift compares each
NOP run to the clean run on the same input and flags when those two tensors
differ even though pytest stayed green.

### Mismatch vs drift

A **mismatch** means the NOP run raised a Python assertion or called
`pytest.fail()`, often because the test's golden check failed.

A **drift** means the NOP run still passed that golden, but its hardware result
is not the same as the clean run's. Drift checks exact element values. Any
element that moved from clean to NOP counts, even if both tensors would still
pass a loose PCC gate against golden. The test's PCC vs
golden can be 0.99 while clean vs NOP might be much lower, and you would never
see it without drift.

Drift findings go in `report.md` and `failures.jsonl` only. The pytest case
stays green and `focus.sh` still exits 0.

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

Only one process should write a report dir at a time. A new run renames
`failures.jsonl` to `failures.jsonl.prev` and drops old `report.md`,
`skips.jsonl`, and `junit.xml`. Use another `--report-dir` per branch or
experiment.

## Hangs

`focus.sh` cannot replace a hung worker or reset the card. After a hang, reset
with `tt-smi -r` before running more device tests. See FOCUS.md for details.

## Sanity check

If you changed the injector, scanner, or reporting, or you suspect they are
broken, run the bundled self-check from the `ttnop` directory.

```bash
make check
```

This runs `tests/ttnop_check.py` on silicon. It arms real L1 detours on a live
datacopy kernel and checks restoration and selected finding types.

- clean `tti_nop` detours on every scanned site restore without a finding
- a planted `ZEROSRC` filler is classified as `mismatch`
- a `tti_nop` detour plus a 1-bit source poke is classified as `drift`

```bash
make check-hang
```

The hang check passes `--hang` to the same script. A self-jumping cave filler
wedges a core and should be classified as `hang`. Run this only when you can
reset the card afterward. It runs `tt-smi -r` when it finishes.
