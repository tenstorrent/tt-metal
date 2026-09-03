# galaxy-kit — rerun the compiler-vs-hand benchmarks on a Blackhole galaxy, in parallel

Three commands compile everything at home, ship it out, race it on every
chip, and bring back one ledger:

```bash
cd ~/sfpi-uplift/galaxy-kit

# 1. compile + ship (quietbox does ALL compiling; the galaxy only executes)
./stage.sh -t ~/sfpi-uplift/sfpi/build/sfpi \
           -f ~/sfpi-uplift/tt-metal-laneLK/tt_metal/tt-llk \
           -w ~/my-run

# 2. red/green pilot, then the full run (job 75439 = the owner's hold)
./run_bench.sh -j 75439 --pilot exp            # must reproduce the board's exp cell
./run_bench.sh -j 75439 -c all -r 5 -k 8       # 32 chips, 5 reps, 8 chips/row

# 3. pull results + build the ledger
./collect.sh -w ~/my-run
```

Outputs land in the workdir: `REPLICATION-LEDGER.tsv` (every rep),
`REPLICATION-PAIRS.tsv` (per same-chip pair), `REPLICATION-VERDICTS.tsv`
(per op: WIN/PARITY/LOSS at the sweep band ±0.5% + board match).

## Prerequisites
- **Route**: the Mac relay must be awake on the LAN (`ssh mac-relay`
  works), and the Mac's qz agent sock must hold the exabox key
  (`~/.ssh/qz-exabox-agent.sock` on the Mac). The quietbox cannot reach
  exabox any other way. Override with `LK_RELAY` / `LK_LOGIN` /
  `LK_AGENT_SOCK` env vars.
- **A held galaxy**: a Slurm job id holding the node (ask the owner).
  The kit only ever does `srun --overlap --jobid <id>` — it never
  scancels, never releases, never `salloc`s.
- **A tt-llk checkout** at the pin you want to measure, with
  `tests/sfpi` pointing at the toolchain under test (stage.sh verifies
  the link and refuses on mismatch; `--relink-sfpi` for a kit-owned farm).
- Remote side needs only `python3` + PyPI (a venv is reused or built
  automatically); `/data` is used for staging (`-d` to change,
  default `/data/nkapre/craq-laneLK`).

## What the kit measures, and the honesty rules baked in
- Rows = every FINAL-BOARD WIN/PARITY/LOSS op + every blaze variant with a
  raced hand arm + the booked trig licensed pair (`-o op1,op2` to narrow).
  Flags come from the checkout's own `sweep_2x2.py` ON set (never
  hand-copied); pinpair rows keep their declared `pin_flags`.
- **Corr-first**: an arm's perf reps run only after its correctness node
  PASSED on that same chip; failures leave `<arm>-CORR-FAIL.txt`.
- **Same-chip pairs**: sem and hand always run back-to-back on the same
  chip; the chip id and node hostname are recorded per cell.
- **Reps**: 5 solo perf sessions per arm per chip (`-r`); reps are
  expected cycle-identical — the ledger reports per-arm spreads.
- **Distinct chips**: `-k N` measures each row on N different chips
  (work-stealing queue; a worker never takes two copies of one row).
- **No resets**: workers never touch tt-smi (EXABOX §7 wall 2); the
  launcher does exactly one marker-guarded upfront `tt-smi -r`.
- **Not canon**: galaxy cycles are NOT p150-canon. The valid statistic is
  the same-chip sem/hand ratio; the p150 board stays canon.

## Anatomy
- `stage.sh`  spec generation (`lib/gen_spec.py`), one
  `pytest --compile-producer` session per flag/env group, bundle pack
  (farm + ELFs + specs + `worker.py`; runtime `riscv-tt-elf-size`/objdump
  only — no compiler ships), streamed to `/data` through the relay.
- `run_bench.sh`  seeds the queue (`lib/seed.py`) and starts one
  `worker.py` per chip via `srun --overlap` (`lib/galaxy_launch.sh`).
  `--pilot <op>` runs one row on one chip; `--status` shows progress.
- `collect.sh`  streams `results/` home and runs `lib/ledger.py`.
- Everything is resume-safe: re-running any stage skips finished builds,
  finished queue items, and finished sessions.
