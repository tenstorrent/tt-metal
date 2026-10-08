# Physical partial-query launch and profiling recovery, Oct 8 UTC

## Current evidence

- The native 32K/B32 full model finished at 06:30:43 UTC with clean device close
  and three measured repetitions: **250.13 output tok/s per TP4**, 7.817
  tok/s/user, 127.93-ms TPOT. Prefill: **5184.79 input tok/s**, 202.50-s TTFT.
  Its single-step comparison is still running; no paired B32 uplift yet.
- The previous publication and report were pushed at `fba818e6acc` to
  `anatarajan/qwen38-long-context-throughput-20261007`.

## Profiling failure and corrected persistent job

The v2 profiling job stopped before device work: the controller selected
`task/source/scripts/run_safe_pytest.sh`, the Blaze wrapper, which does not
implement `--profile-ops`. Pytest rejected that argument. Retained original
log and failed queue receipt are included here; no hardware failure or new
profile measurement is implied.

The controller now uses a profiling-capable Metal wrapper frozen with its
source and included in source-hash checks. Before waiting for the device,
it validates the option, checks shell syntax and verifies `python -m tracy
--help`. No runtime/install source is modified. The exact frozen wrapper,
Tracy preflight output, launch/source hashes and CPU JUnit are included.

`qwen38-bounded-layer-profile-v3-20261008.service` launched at 06:35:55 UTC
after **276 CPU tests plus 40 subtests** passed. Order: 32K/B16/B32,
16K/B16/B32, 128K/B16 and 256K/B8, native and single-step for each: **12
captures**. Each uses two real layers with seeded synthetic caches, two warm
calls and one measured call; it does not do prefill or establish traced
full-model throughput. Per-op timings include waits. Full profiling gate
and reference-eval qualification remain open.

## Partial-query physical sweep

`qwen38-attention-half-tile-hardware-v1-20261008.service` launched at
06:40:24 UTC after **295 CPU tests plus 40 subtests** passed. It independently
revalidated the completed simulator evidence and matched the exact candidate
compute header before entering the hardware queue.

There are **10 geometries and 30 component cases**, ordered:

1. 32K at B8/B16/B32.
2. 16K at B8/B16/B32.
3. 128K at B8/B16; near-256K at B4/B8.

Each partial-query case is bracketed by full-query controls. Each case has
five samples of 100 trace replays, then repeats the measurement. All four
ranks must pass the existing per-user PCC/RMS checks; input/reference hashes
must match; unstable replay or >3% timing drift cannot yield a qualified
speedup. Numerical failures remain visible and excluded rather than aborting
before other geometries run. A completed diagnostic is not proof that every
candidate passed; consult `all_candidates_qualified` and each comparison.

The full-query controls use unchanged full-tile arithmetic in the same source
overlay. The experiment runs production instructions without the simulator's
SFPLOADMACRO fallback and checks the generated compute includes. It measures
the SDPA kernel boundary, excluding model padding/slicing operations. Partial
query tiles also affect native factory buffers and reader thresholds, so any
gain is not an isolated exp-instruction effect. It does not reduce KV bytes.

Hardware accuracy, timing and full-model gains are still unproven while queued.
The candidate is not enabled in model or serving defaults.

## Persistence, resource limits and scope of estimates

Collection-time systemd evidence confirms the full-model, v3 profile and
partial-query controllers live; `loginctl` confirms `Linger=yes`. They survive
SSH/client disconnects and save logs/results to the allocated host disk.
All hardware access shares `/tmp/tt-device.lock`; no concurrent device use.
These services do not automatically resume after a host reboot.

- Full-model controller: 14-hour outer deadline. Three runs remained at the
  count check: 32K/B32 single-step, 256K/B8 native and single-step. Roughly
  1.5-2 hours of model work, plus diagnostic contention; not a completion SLA.
- Profile controller: 12-hour outer deadline, 64 GiB/eight CPUs; 1-GiB file,
  4-GiB capture limits. First successful bounded capture is still pending, so
  a reliable total profiling ETA is not established.
- Partial-query controller: 5-hour outer deadline, 48 GiB/eight CPUs; the
  hardware child has a 3-hour deadline including lock wait and a 45-minute
  pytest limit after starting. These limits are timeouts, not expected runtime.

Priority remains 32K first, 16K second, with active 128K/256K optimization.
No precision reduction, installation change, checkpoint edit or firmware change.
