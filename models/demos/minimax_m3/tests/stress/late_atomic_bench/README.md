# Late atomic response vs the next kernel's NoC counter snapshot

Single-chip microbenchmark, part of the [M3 prefill stress test](../README.md). It tests whether an NCRISC kernel that exits with a non-posted atomic still
in flight can wedge the next NCRISC kernel's `noc_async_atomic_barrier()` on Blackhole.

## Mechanism under test

- The NCRISC kernel prologue (`tt_metal/hw/firmware/src/tt-1xx/ncrisck.cc`) calls `noc_local_state_init(NOC_INDEX)`
  before `wait_for_go_message()`. That copies `NIU_MST_ATOMIC_RESP_RECEIVED` into `noc_nonposted_atomics_acked`.
- Fast dispatch always sets `DISPATCH_ENABLE_FLAG_PRELOAD` (`tt_metal/impl/program/dispatch.cpp:2320`). So once a
  core finishes kernel A, BRISC starts B's firmware setup right away (`brisc.cc` go loop). B's NCRISC prologue
  takes the snapshot a few firmware steps after A returns, before the dispatcher has collected A's completions.
- If the response to A's last atomic lands after that snapshot, hw = sw + 1 for all of B.
  `ncrisc_noc_nonposted_atomics_flushed` (`noc_nonblocking_api.h:760`) tests exact equality.
- Real instance: deepseek_prefill `reader_worker_dispatch.cpp` does `next_turn_sem.up(...)` and then exits. The fix
  waits for that atomic before the kernel returns (`noc.async_atomic_barrier()` at the end of `kernel_main`, PR #60555).

### When does an off-by-one turn into a hang?

The sw counter goes up by one when an atomic is issued. The hw counter goes up by one when a response
arrives. With one extra response counted (hw = sw + 1 when idle), a barrier that follows k fresh atomics
returns at the moment k-1 of them have been answered:

| B pattern | result with hw = sw + 1 |
|---|---|
| 1 inc, then barrier right away (`inc_barrier`) | **never hangs.** It returns early, before its own response arrives, and leaves hw = sw + 1 (counted as `resid`). |
| k >= 2 incs, then barrier (`incn_barrier`, the moe_fused_swiglu invite loops) | hangs only if the polling loop misses the short window between the last two responses |
| barrier with no own atomic in flight: inc, other work, barrier (`late_barrier`), or `noc_async_full_barrier()` (`full_barrier`) | **always hangs** |

So a "one inc + barrier" B cannot show a hang by design. The bench runs it to count early
returns, and uses `late_barrier` / `full_barrier` as the deterministic hang detectors.

Side note: the BH NoC is a unidirectional torus per NoC, and responses travel on the same NoC. The round
trip therefore covers the full ring in each dimension the request crosses, whichever target you pick. A
"far" target only helps because it puts the atomic behind the flood traffic. Distance alone does not help.

## Layout

| Program | Cores | RISC / NoC | Kernel |
|---|---|---|---|
| A | A cores = logical column 0 (`--a-cores N`, default the whole column) | NCRISC, NOC`--noc` (default 1) | `kernels/late_atomic_a.cpp`: wait `--a-delay` + pseudo-random `0..--a-jitter` cycles, `--a-incs` x `noc_semaphore_inc` to the far core, exit (`--a-barrier`: wait for the responses first and record the RTT) |
| A | every other core except the far one (congestors) | BRISC, same NoC | `kernels/late_atomic_flood.cpp`: for `--flood-cycles`, stream `--chunk-bytes` writes (`--flood-kind write`), atomics (`atomic`) or both, then drain. `--flood-dest far` floods the far core (request path), `acores` floods the A cores (response path), `both` alternates |
| B | A cores | NCRISC, same NoC | `kernels/late_atomic_b.cpp`: probe hw - sw at kernel_main entry (d0) and after `--probe-wait` cycles (d1), then run the `--b-mode` action |

The far core is logical (gx-1, gy-1) and runs no kernel. All state lives in a height-sharded L1 tensor
that spans the whole grid, so each address is the same on every core. The last 4 KB row of each core holds
the stats words (`kernels/late_atomic_common.h`). The host reads them through the CQ while things run
normally. After a hang it reads them with `ttnn.cluster.read_from_core` (UMD path).

The host warms up with one eager A,B pair. It then captures `--trace-pairs` (256) A,B pairs into one trace
and replays it with no host sync in between until `--pairs` is reached. It syncs every `--sync-pairs`.

## Prerequisites

- A free chip. The bench uses one chip, no fabric. Run nothing else on it.
- A built checkout (`./build_metal.sh`). The kernels are JIT-compiled from `kernels/`, so nothing else needs building.
- Watcher off. With watcher or lightweight asserts on, A trips `DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped`
  at its first exit. That is a quick check that A leaves the atomic in flight, but it is not this experiment.

```bash
cd $TT_METAL_HOME && source python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME TT_VISIBLE_DEVICES=0
B=models/demos/minimax_m3/tests/stress/late_atomic_bench
```

## Run

One config (the main measurement; it cannot hang):

```bash
python $B/late_atomic_bench.py --b-mode probe
```

The whole matrix, from safest to most hang-prone. It stops at the first hang because the chip then needs
a reset:

```bash
TT_VISIBLE_DEVICES=0 $B/run_matrix.sh
# Continue past hangs (resets the WHOLE galaxy; only when you hold the box):
RESET_CMD="env -u TT_VISIBLE_DEVICES tt-smi -glx_reset" TT_VISIBLE_DEVICES=0 $B/run_matrix.sh
# Subset / shorter:
PAIRS=200000 ONLY="probe_flood ctl_late late_barrier" $B/run_matrix.sh
```

Outputs go to `$TT_METAL_HOME/generated/late_atomic_bench/<utc>_<tag>/` (`log.txt`, `summary.json`, and
`triage_on_timeout.txt` on a hang). The matrix writes to `generated/late_atomic_bench/matrix_<utc>/`.

Useful knobs: `--a-incs 16` (more responses in flight at exit), `--flood-kind both`, `--flood-cycles`,
`--a-delay/--a-jitter` (where in the flood A fires), `--noc 0`, `--trace-pairs 0` (eager enqueue; slower,
host-bound), `--hold-on-hang` (keep the hung process alive for manual tt-triage).

## Reading the output

Progress lines and `SUMMARY` hold the counts summed over the A cores:

- `b_d1_nz`: number of B runs that saw hw != sw after the probe wait. **This is the key metric.** Anything
  above 0 means a response to A arrived after B's prologue snapshot, i.e. the race is real on silicon.
  `b_dmax` is the largest difference seen (1 per late response).
- `b_resid_nz`: B's barrier returned, but hw != sw afterwards. This is the early return of `inc_barrier`.
- `b_gap_min/max`: cycles from A's exit to B's `kernel_main` on the same core. This includes waiting for
  the go signal, so it is an upper bound on the snapshot window.
- Control only (`--a-barrier`): `a_bar_max/a_bar_mean`, and `a_bar_gt_<thresh>` = how many A responses took
  longer than `--late-thresh` cycles under this flood. If the control never shows RTTs above roughly
  1-3k cycles (rough estimate of A exit -> B snapshot), the no-barrier runs will not see late responses.
  In that case make the flood heavier rather than running longer.

On a hang (exit code 3), the log lists each A core: B state (`IN-BARRIER` means stuck in B's action),
hw/sw at action entry, and `diff`. A core stuck in the barrier with diff 1 confirms the mechanism. The
dispatch timeout handler runs tt-triage on its own (`triage_on_timeout.txt`: `dump_running_operations` +
`dump_callstacks -vv`). Expect the A cores' NCRISC in `noc_async_atomic_barrier` (waypoint `NABW`,
`late_atomic_b.cpp:79/87/91`) or `noc_async_full_barrier` (waypoint `NFEW`, line 93). Manual check with
`--hold-on-hang`, from a second shell while the bench process is still alive:

```bash
cd $TT_METAL_HOME && TT_VISIBLE_DEVICES=0 python3 tools/tt-triage.py --disable-progress \
  --run=dump_callstacks --run=dump_risc_debug_signals -vv
```

Then kill the bench and reset (`env -u TT_VISIBLE_DEVICES tt-smi -glx_reset` on the galaxy).

## Expected outcomes

| matrix name | args | expectation if the mechanism is real | if it is not |
|---|---|---|---|
| `noflood_probe` | `--flood-kind none --b-mode probe` | `b_d1_nz` = 0: an unloaded response beats the firmware handoff | same |
| `probe_flood` | `--b-mode probe` | `b_d1_nz` > 0, never hangs | `b_d1_nz` = 0 |
| `probe_flood_x16` | `--b-mode probe --a-incs 16 --flood-kind both` | more hits; `b_dmax` can exceed 1 | 0 |
| `ctl_late` | `--a-barrier --b-mode late_barrier` | no hang, `b_d1_nz` = 0; `a_bar_*` gives the congested RTT | same |
| `ctl_full_x16` | `--a-barrier --b-mode full_barrier --a-incs 16 --flood-kind both` | no hang, `b_d1_nz` = 0 | same |
| `inc_barrier` | `--b-mode inc_barrier` | no hang; `b_resid_nz` is about `b_d1_nz` (early returns) | 0/0 |
| `incn_barrier` | `--b-mode incn_barrier --b-incs 4` | may hang; per-event odds depend on response spacing vs the polling loop | no hang |
| `late_barrier` | `--b-mode late_barrier` | **hangs at the first late response** (B core IN-BARRIER, diff 1) | completes |
| `full_barrier_x16` | `--b-mode full_barrier --a-incs 16 --flood-kind both` | **hangs** (waypoint NFEW) | completes |
| `late_barrier_noc0` | `--b-mode late_barrier --noc 0` | same as `late_barrier`, on NoC 0 | completes |

The controls must never hang and must show `b_d1_nz` = 0. If they do not, something other than A's
missing barrier is moving the counter, and the bench is not valid.

## Measured on a Blackhole galaxy

300k A,B pairs per config at about 7.5k pairs/s (NoC 1 unless noted). One build without the fix above.

| config | chip 0 | chips 17, 30 |
|---|---|---|
| `noflood_probe` | `b_d1_nz` 0 of 3.03M | |
| `probe_flood` | `b_d1_nz` 2.85M of 3.03M (94%), `b_dmax` 1 | 94% |
| `probe_flood_x16` | `b_d1_nz` 3.03M of 3.03M, `b_dmax` 16 | |
| `ctl_late` | no hang, `b_d1_nz` 0; `a_bar` max 6369, mean 1527 cycles; 1.52M responses > 1500 cycles | |
| `ctl_full_x16` | no hang, `b_d1_nz` 0; `a_bar` max 25345, mean 15293 cycles | |
| `inc_barrier` | no hang, `b_resid_nz` 2.85M (every late response gives an early return) | |
| `incn_barrier` | hang in the first trace (one core IN-BARRIER, diff 1) | hang within 80 / 200 B runs |
| `late_barrier`, `full_barrier_x16`, `late_barrier_noc0` | hang in the first trace (all A cores IN-BARRIER) | |

The controls are clean everywhere. Under NoC load, an NCRISC kernel that exits with a non-posted atomic in flight
leaks the response into the next NCRISC kernel's counters about 94% of the time. A following multi-inc + atomic
barrier (the moe_fused_swiglu invite pattern), or any barrier with no fresh atomic, then hangs.

Each hang costs a dispatch timeout (30 s), the tt-triage run (1-3 min) and a reset. Change `--pairs` to fit the slot.
