# Hangs: prevention, detection, recovery

Read before the first device run of any session. A hang burns the tool timeout,
leaves the device dirty, and makes the *next* run fail somewhere unrelated —
which reads as a new bug and sends you debugging the wrong thing.

## Three rules

**1. Every run is timeout-gated.** No exception for "this one is quick".

```bash
timeout 900 ./python_env/bin/python -m pytest <path> -k <filter> -x -s --timeout 600 &> run.log
```

Or in the test: `@pytest.mark.timeout(3600)` (used in `tests/models/ltx/`).
`--timeout=0` is reserved for sweeps that bound themselves per configuration
(`tests/models/wan2_2/bruteforce_conv3d_sweep.py`); the inner loop must
then have the timeout. **Never pipe a device run to `tail -N`** — buffering keeps
the log empty until exit, so a hang shows you nothing.

**2. Bounded polling with an early bail.** Grep for failure signatures each
iteration, not just success.

```bash
for i in $(seq 1 30); do
  pgrep -f "[p]ytest.*<filter>" >/dev/null || break
  grep -qE "TT_FATAL|TT_THROW|Traceback|Watcher detected|[0-9]+ (passed|failed) in " run.log && break
  sleep 10
done
echo "waited ${i}0s; alive=$(pgrep -cf '[p]ytest.*<filter>')"
```

Size the cap from the expected runtime — a 70 s test gets ~2 min, not 10. Match
pytest's *summary* line; sweep scripts print their own `21 ok, 43 failed`
progress lines that trip looser patterns.

**3. Reset after every kill, before concluding anything.**

```bash
fuser -v /dev/tenstorrent/* 2>&1 | head          # who still holds the device
kill -9 -- -$(ps -o pgid= <pid> | tr -d ' ')     # process GROUP, not just the PID
tt-smi -glx_reset
tt-smi -ls                                        # verify it comes back
```

| Rule | Why |
|---|---|
| Kill the stale holder first | UMD takes a named mutex (`/dev/shm/TT_UMD_LOCK.CHIP_IN_USE_<n>_PCIe`) on device open; a hung run keeps holding it and the device will not initialise until the holder dies |
| Kill the process **group** | A `tracy` run spawns children; any one of them can be the holder |
| Don't delete `/dev/shm/TT_UMD_LOCK.*` | Named-mutex backing files persist normally. The problem is the live process, not the file |
| **`tt-smi -r` is forbidden on CPLD < 1.16** | It dropped all 32 chips off the PCIe bus and required a host reboot |

## Detection

```bash
TT_METAL_WATCHER=10 ./python_env/bin/python -m pytest <path> -s --timeout 600 &> run.log
```

Watcher monitors firmware and kernels for NoC errors, asserts and stalls. You
should see `Watcher checking device <n>`; on a catch it prints the error, the
core, the last waypoint, and the running kernels. Log:
`generated/watcher/watcher.log` (legend at the top).

| Field | Reads as |
|---|---|
| `k_ids:<brisc>\|<ncrisc>\|<trisc>` | Which kernels were on the core, mapped to filenames at the end of the dump — this is what to debug |
| waypoint (`GW`, `NARW`, `NTW`, …) | Last point each RISC passed. `W` = waiting; a core stuck in a wait waypoint is the hang |

| Situation | Tool |
|---|---|
| Expecting trouble | `TT_METAL_WATCHER=10` up front |
| Hang only reproduces **without** watcher (timing-sensitive) | Peel back the most invasive feature: `TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1`, then `..._DISABLE_WAYPOINT=1` |
| Run wasn't watched, device already wedged | `./build/tools/watcher_dump --devices=<ids>` then read the log. Needs the PCIe/ethernet link still up |
| Know *which* core is stuck, not *where* in the kernel | `./tools/tt-triage.py --run=dump_callstacks` (add `--all-cores -vv`). Wraps `tt-exalens`; gives kernel ID/name, go message, waypoint, PC and callstack per core |

**Watcher conflicts with the profiler and DPRINT** — `TT_METAL_WATCHER`,
`TT_METAL_DEVICE_PROFILER` and `TT_METAL_DPRINT_CORES` all use device SRAM.
Profile *or* watch, never both.

## Escalation order

1. Timeout fires → kill, capture the log tail, kill any stale chip holder, `tt-smi -glx_reset`.
2. Reproduce with `TT_METAL_WATCHER=10`. Most hangs are named outright.
3. Watcher clean but still hanging → read `watcher.log` for `k_ids` and waypoints.
4. Need the kernel-level position → `tt-triage.py --run=dump_callstacks`.
5. Cannot re-run with watcher → `watcher_dump` post-hoc.

Record the hang, cause and fix in the journal's `Hangs / resets` section. Repeat
hangs on the same shape are a design signal — see `known-issues.md`.

## Sweep discipline

Sweeps are the most reliable way to find a hang, so bound them structurally:

- **One configuration per process**, hard per-config timeout. A combined loop
  that hangs on config 3 costs you configs 4–20 as well.
- **Treat unmeasured configurations as suspect, not absent.** When a sweep hangs
  partway the honest record is "best of what ran", not "optimal". Note in the
  journal which values were never reached.

Per-process isolation bounds the damage but does not prevent the hang: an SDPA
hang reproduced with each config in its own process, localising it to the
configuration rather than to sweeping.
