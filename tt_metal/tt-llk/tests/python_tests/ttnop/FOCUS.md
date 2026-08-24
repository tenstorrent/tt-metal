# focus.sh — depth runs on one test case

`focus.sh` takes **one** pytest node id and re-runs it many times with a delay injected at a chosen point, so a race that fails intermittently comes back as a rate instead of a pass/fail coin flip. Everything is a knob, so you can go from "every sync site in this test" down to "this one `SEMGET`, with this
one nop, at exactly 54 counts".

```bash
./focus.sh 'test_eltwise_unary_datacopy.py::test_unary_datacopy[formats:Float16_b->Float16_b-dest_acc:No-num_faces:1-tilize:No-input_dimensions:[64, 64]]'

./focus.sh --sites unpack:3 --nop risc_nop --delays 8,16 \
    'test_mul_reduce_scalar.py::test_mul_reduce_scalar[formats:Float16_b->Float16_b-math_fidelity:HiFi2-num_tiles:1-tile_dimensions:[16, 16]]'
```

Every flag is an alias for the `TTNOP_*` variable of the same name, so the two
styles are interchangeable and a flag wins over an inherited export:

```bash
TTNOP_SITES=unpack:3 TTNOP_DELAYS=8,16 ./focus.sh \
    'test_mul_reduce_scalar.py::test_mul_reduce_scalar[formats:Float16_b->Float16_b-math_fidelity:HiFi2-num_tiles:1-tile_dimensions:[16, 16]]'
```

Quote the node id: the bracketed params usually contain characters the shell
would eat, and a split id is rejected rather than half-run.

## What one invocation does

1. Compiles just this case into the shared build tree (skipped if already there).
2. Runs the case **clean** once. That pass is the baseline: it loads the ELFs,
  draws the stimuli, and produces the output every variant is compared against.
3. Scans each thread ELF for sites and expands the knobs below into a variant
  list: `(thread, site, nop type, count)`, of which this worker takes its share.
4. Runs each variant `TTNOP_REPEATS` times, counting failures. A hang is
  soft-reset and the sweep continues; if the reset does not take, the finding is
   recorded and the sweep stops rather than reporting a dead card as a wall of races.
5. Writes `failures.jsonl` and renders `report.md` from it.



## The four axes you can narrow


| flag       | variable          | default          | what it selects                                                                                                                 |
| ---------- | ----------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `--thread` | `TTNOP_THREADS`   | `unpack,math`    | which thread's ELF gets patched; add `pack` for all three                                                                       |
| `--site`   | `TTNOP_SITE_MODE` | `sync`           | `sync` for sync/stall instructions only. `all` for every candidate instruction takes a long time, a kernel has hundreds of them |
| `--sites`  | `TTNOP_SITES`     | every site found | `unpack:3,math:7` — site index per thread                                                                                       |
| `--nop`    | `TTNOP_FILLER`    | `auto`           | `tti_nop`, `risc_nop`, `sfpnop`, `unpacr0`, `unpacr1`, a raw hex word, or `auto`                                                |
| `--delays` | `TTNOP_DELAYS`    | `1-100`          | how many nops: ints and `lo-hi` ranges, e.g. `1-8,16,32`                                                                        |


`auto` tries `tti_nop` and `risc_nop` at every site, plus whichever unit-retired
nop applies (`unpacr0`/`unpacr1` on unpack sync sites, `sfpnop` on SFPU sites).
Naming one nop — `--nop risc_nop` — is the fastest way to cut a run down, since
the named nops perturb in different directions and usually only one of them
opens the window.

## The rest of the knobs


| flag            | variable            | default         | meaning                                                                                                                                                              |
| --------------- | ------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--repeats`     | `TTNOP_REPEATS`     | `10`            | runs per variant. This is the denominator of the rate                                                                                                                |
| `--device-jobs` | `TTNOP_DEVICE_JOBS` | `8`             | Tensix cores to spread the sweep over; `1` runs it in this process                                                                                                   |
| `--max-delay`   | `TTNOP_MAX_DELAY`   | `100`           | cave capacity; raise it to sweep counts above 100. Too big and there may not be enough free space after `_etext` (before the next L1 section) to hold the filler run |
| `--no-drift`    | `TTNOP_DRIFT`       | on              | drift freezes stimuli and compares each run to the clean one; off rolls the RNG instead                                                                              |
| `--report-dir`  | `TTNOP_REPORT_DIR`  | `reports/focus` | where `failures.jsonl` and `report.md` land                                                                                                                          |
| `--verbose`     | `TTNOP_VERBOSE`     | off             | print every detour as it is armed                                                                                                                                    |
|                 | `CHIP_ARCH`         | `wormhole`      | picks the build and the device lock                                                                                                                                  |


Any `TTNOP_REPEATS > 1` also adds a **delay-0 control** per site and nop: it
still detours through the cave but executes no fillers. If 0 passes and 54 fails,
the fillers did it and not the jump.

## Budget it first

Example: `test_mul_reduce_scalar` has a few hundred instructions per thread and
a handful of sync sites. At the defaults (`auto` ≈ 3 nops at an unpack site,
delays 1-100 plus a delay-0 control, 10 repeats) one of those sites is already
3 × 101 × 10 = 3030 variants, and each is a full test body. `--site all` uses
the instruction count as the site count, so the same test becomes tens of
thousands. Narrow at least two axes:


| invocation                                       | runs | per core at `--device-jobs 8` |
| ------------------------------------------------ | ---- | ----------------------------- |
| `--sites unpack:3 --nop risc_nop --delays 54`    | 20   | 3                             |
| `--sites unpack:3 --nop risc_nop --delays 40-60` | 220  | 28                            |
| `--sites unpack:3 --delays 1-100` (3 nop types)  | 3030 | 379                           |


The `Reproduce` block of any `report.md` is already a narrowed `focus.sh` line
for one site, so the usual workflow is to copy one out rather than write it.

## Reading the report

`failures.jsonl` is the source of truth (one line per recorded variant, appended
as the sweep goes, so it survives a kill). `report.md` is the view, and per site
it gives you:

- **Sites** — one row per site and nop type: which counts failed, folded into
ranges, and how (`mismatch`, `hang`, `assert`, `error`, `drift`).
- **NOP types** — the same for one site, with the exact filler word.
- **Failure rate** — `fails / runs` per count. The reason to use this script.
- **Where the NOPs went in** — the `addr2line` inline chain for the patched
address, innermost frame first, resolved while the ELF still exists.
- **Reproduce** — a copy-paste line that re-runs just those counts.

A **mismatch** is the delayed run failing the test's own golden. A **drift** is
it still passing the golden but not matching the clean run bit for bit; that is
report-only, and keeps the case green. Every recorded variant is listed (one row
per site and nop type). Counts that passed are not in the log.

## Exit codes and gotchas

- `0` clean, or drift only. `1` at least one variant broke the golden. `4` bad usage.
- Each run starts on an empty log: the last `failures.jsonl` is moved aside to
`failures.jsonl.prev` and a stale `report.md` is removed. Only one generation
is kept, so use `--report-dir` for anything you want to hold on to.
- No watchdog: a soft reset per hang is the only recovery, so a case that hangs on
most of its variants is slow however many cores it runs on.
- All the workers append to one `failures.jsonl`. Records carry their plan
position, so the report still reads in sweep order.
- The build and device locks (`TTNOP_BUILD_LOCK`, `TTNOP_DEVICE_LOCK`) are shared
with every other runner on the host, so an invocation may wait before it starts.
- Delays above `TTNOP_MAX_DELAY` are rejected up front, not clamped.
