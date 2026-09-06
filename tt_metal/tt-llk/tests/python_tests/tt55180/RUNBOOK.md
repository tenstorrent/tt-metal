# #55180 — reproducer runbook

One command, two arms.

```bash
cd tt_metal/tt-llk/tests/python_tests/tt55180
./run_55180.sh                 # silicon (Blackhole)
./run_55180.sh --sim           # simulator (see below)
```

Output lands in `./results/`: `report.txt` plus a `.log` per run.

## What it does

| arm | defines | expected |
| --- | --- | --- |
| `ARM_FIX` | `TT55180_REPRO_NOPS=6 TT55180_FIX_UNPACK_STALL=1` | run1 pass, run2 **pass** |
| `ARM_HANG` | `TT55180_REPRO_NOPS=6 TT55180_FIX_UNPACK_STALL=0` | run1 pass, run2 **hang** |

The fix arm runs first and is timed. The hang arm then gets a per-run budget of
**2× the fix arm's slowest run**; exceeding it is itself the hang signal, so the
verdict never depends on parsing a harness message.

## The one thing that will catch you out

**The hang is on the SECOND run after a device reset.** Run 1 always passes, and a
reset puts the device back to the state where it passes. So a **single run from a
cold device cannot reproduce this**. On silicon the script handles it; on a
simulator you must execute the kernel **twice inside one session**, without
restarting in between — a restart behaves like a reset.

## The two knobs

Both live in `tt_llk_blackhole/llk_lib/llk_unpack_A.h` and are **OFF by default**,
so the branch is inert unless you pass the defines.

- `TT55180_REPRO_NOPS=N` — N pure `UNP_NOP` words on unpacker 1 before the stall.
  Measured over N=0…20: **only 6 and 7 reproduce**.
- `TT55180_FIX_UNPACK_STALL=1` — also wait on the unpackers at that stall. Makes the
  reproducer pass.

The script validates both before each arm: the ELF must contain 6 NOP words and the
expected stall encoding, otherwise it aborts rather than reporting a false result.

## First-time setup

```bash
bash tt_metal/tt-llk/tests/setup_testing_env.sh   # fetches this branch's pinned sfpi
```

Do not reuse another worktree's `tests/sfpi` — the pins differ between branches and a
mismatch fails the build with confusing template errors.

## On a simulator

`--sim` skips the device reset and says so. You still need to:

1. Start the simulator server first and leave it up for **both** runs of an arm.
2. **Set the simulation timeout explicitly for the hang arm.** The default is far
   larger than a hanging run will ever reach in practical wall clock. Size it from
   the fix arm's reported finish time — ~2× is the same budget rule this script uses
   on silicon. Mind the units: logs and the plusarg differ.
3. Shut the server down cleanly. A hard kill can invalidate captured output.

## Optional state capture

Set `TT55180_DUMP_HOOK` to an executable of your own to snapshot device state at each
step; it is called as `$TT55180_DUMP_HOOK <outfile> <core>`. Unset, the step is
skipped. Override the core with `CORE=...` (the harness location format, e.g. `0,0`).

## Status

- Measured on Blackhole: the fix arm passes both runs; the unfixed arm reproduces on
  run 2 at N=6 and N=7.
- **This is a reproducer, not a demonstration that shipped code fails.** It needs
  build-time NOPs that do not exist in the shipped kernel. Whether the shipped kernel
  can reach the same window unaided is open, and gates any fix PR.
