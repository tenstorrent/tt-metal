# Handoff: intermittent mid-denoise device hang (MiniMax-H3 t2va, Wormhole 4x8)

Observed 2026-09-16 while sweeping `test_t2va_end_to_end` on a Wormhole galaxy
(4x8, 32 chips). **This is not a memory bug** — it appeared only after DiT FSDP
removed the OOM ceiling and long runs actually got to execute. It is now the
blocker for the last 5 of 18 working points.

**I do not have a deterministic reproducer.** 2 hangs in 12 completed long runs.
What follows is the exact fingerprint, the observed conditions, and the cheapest
path to another instance.

Raw logs are not committed; on the run host they are `~/h3_wormhole_results/sweep_fsdp.log.gz`
(hang 1) and `sweep15b.log.gz` (hang 2).

## Symptom fingerprint

Use this to confirm you have *this* bug and not an OOM or a crash:

- Test produces **no further log output**, indefinitely. No exception, no traceback.
- Process is **spinning, not idle**: 171–316% CPU, and CPU-time climbs well past
  elapsed time (measured 06:23:27 CPU over 02:01:14 elapsed).
- Every thread sits in `futex_wait_queue`:
  ```bash
  P=$(pgrep -f "^python -m pytest models/tt_dit" | head -1)
  ps -o pid,stat,etime,time,wchan:24,pcpu -p $P
  for t in /proc/$P/task/*; do echo "$(basename $t) $(cat $t/wchan)"; done | sort | uniq -c
  ```
  That is tt-metal's host dispatch busy-polling a device that stopped retiring work.
- **`@pytest.mark.timeout(7200)` does not fire.** pytest-timeout cannot interrupt a
  C-level stall, so the test hangs past its own deadline. Wrap runs in `timeout(1)`
  at the shell level — that is what eventually ended both of these.
- Afterwards the **board is wedged**. The next process to open a device fails at
  setup with:
  ```
  Device 9 init: failed to initialize FW! Try resetting the board.
  RuntimeError: TT_THROW @ tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:1573
  ```
  In the sweep this surfaced as `6 errors` at fixture setup, not as test failures.

## The two observed instances

| | hang 1 | hang 2 |
|---|---|---|
| case | `9x16_10s` (768x1344, 243 frames) | `16x9_15s` (1344x768, 362 frames) |
| which generation | **timed** (2nd pass) | **warmup** (1st pass) |
| last logged step | `step 21/49 t=0.0543` | `step 21/49 t=0.0543` |
| time of that log | 08:56:22.034 | 10:05:17.577 |
| last line in log | 08:56:35.503 | 10:05:30.603 |
| **delta after step 21** | **13.47 s** | **13.03 s** |
| per-step rate | ~6.6 s/step | ~12.7 s/step |
| implied stall point | ~step 23 | ~step 22 |
| cases completed before it | 11 | 1 |
| elapsed load before it | ~1 h 27 m | ~30 m (devices freshly reset) |
| DiT FSDP | on | on |

### What the timings suggest

Both stalled **~13 s after logging step 21**, despite the two cases running at very
different per-step rates (6.6 vs 12.7 s/step). If the trigger were step-aligned you
would expect the wall-clock offsets to differ; they agree to within half a second.
Two samples is not enough to claim a cause — record the offset on any new instance,
it is the most discriminating number available.

(Step logging is every 10 steps, so "last logged step 21" only bounds the stall to
steps 21–30; the +13 s offset narrows it to ~step 22–23.)

## How to attempt a reproduction

Ordered cheapest-first. **Reset the boards before each attempt** (see Recovery) so
you are not measuring leftover state.

### 1. Re-run the two known cases (~30 min each, best odds)

```bash
TT_DIT_CACHE_DIR=~/tt_dit_cache MINIMAX_H3_DIT_FSDP=1 RUN_VBENCH=0 \
  timeout 3600 python -m pytest \
  models/tt_dit/tests/models/minimax_h3/test_pipeline_minimax_h3.py \
  -k "4x8nl4 and 9x16_10s" -q 2>&1 | tee ~/hang-9x16-10s.log
```

then the same with `-k "4x8nl4 and 16x9_15s"`.

This is the single most valuable experiment: it separates **deterministic**
(these working points always hang) from **flaky** (they were unlucky). That
distinction decides whether the remaining 5 points are reachable at all, and it
is cheap. Note hang 2 occurred only ~30 min after a fresh reset on the 2nd case,
which already argues against a pure accumulation/thermal explanation.

### 2. Loop one long case (unattended, highest cumulative odds)

```bash
for i in $(seq 1 6); do
  echo "=== iteration $i $(date) ==="
  TT_DIT_CACHE_DIR=~/tt_dit_cache MINIMAX_H3_DIT_FSDP=1 RUN_VBENCH=0 \
    timeout 2400 python -m pytest \
    models/tt_dit/tests/models/minimax_h3/test_pipeline_minimax_h3.py \
    -k "4x8nl4 and 16x9_10s" -q
  echo "exit=$?"
done 2>&1 | tee ~/hang-loop.log
```

`16x9_10s` passed cleanly before, so a hang here proves the bug is not tied to a
particular working point. Watch for a `timeout`-induced exit 124 — that is a hang,
and it will wedge the boards, so the following iterations will error at setup.
Expect to reset between iterations if one hangs.

### 3. The full sweep (~4 h, reproduces the original conditions)

```bash
TT_DIT_CACHE_DIR=~/tt_dit_cache MINIMAX_H3_DIT_FSDP=1 RUN_VBENCH=0 \
  timeout 36000 python -m pytest \
  models/tt_dit/tests/models/minimax_h3/test_pipeline_minimax_h3.py \
  -k "4x8nl4" -q 2>&1 | tee ~/hang-sweep.log
```

This is how both hangs were found, but one hang stops all later cases, so it is a
poor instrument for characterising the bug.

## Recovery

```bash
tt-smi -glx_reset          # USER_RESET on 32 devices -> IPMI -> POST_RESET, ~7 min
```

tt-smi prints a hint that plain `tt-smi -r` is also supported on Galaxy 6U.
After it completes you should see `Re-initialized 32 boards after reset`.

Before resetting, confirm nobody else is on the box — this is destructive to any
concurrent user:

```bash
for p in $(ls /proc | grep -E '^[0-9]+$'); do
  ls -l /proc/$p/fd 2>/dev/null | grep -q tenstorrent && \
    echo "pid $p user=$(stat -c %U /proc/$p) cmd=$(tr '\0' ' ' </proc/$p/cmdline | cut -c1-60)"
done
```

Verify health afterwards:

```bash
python -c "import ttnn; d=ttnn.open_mesh_device(ttnn.MeshShape(1,1)); print('OK'); ttnn.close_mesh_device(d)"
```

A graceful `kill -TERM` on the hung pytest **does** exit cleanly, but does **not**
un-wedge the board — hang 1 was SIGTERMed successfully and the next run still hit
`failed to initialize FW`.

## What is ruled out

- **Not memory.** Both hangs happened with FSDP on, where DiT residency is
  101.8 MiB/bank of 1021 and free space is ~919 MiB/bank. No OOM is logged, and
  OOMs in this codebase raise `TT_FATAL ... Out of Memory` promptly rather than stalling.
- **Not canvas-specific.** `21x9_15s` (1536x672, 1.03 MPix) passed in the same run
  where `16x9_15s` (1344x768, 1.03 MPix) hung — same pixel count, same duration.
- **Not purely duration.** `16x9_10s` and `21x9_10s` both passed; `9x16_10s` hung.
- **Not accumulated load alone.** Hang 2 came 30 min after a full galaxy reset,
  on only the 2nd case of the run.
- **Not the corrupt weight cache** that caused a separate wrong-output bug earlier
  (see `MiniMaxH3_wormhole_perf.md` open issue 3) — `~/tt_dit_cache` was rebuilt clean before these
  runs and CLIP matched baseline exactly on all six 5 s cases.

## Possibly the same underlying bug

Earlier the same box produced **two hard SIGBUS crashes** in the denoise loop, with
FSDP *off*:

```
Fatal Python error: Bus error
  models/tt_dit/utils/tensor.py:349 in local_device_to_torch
  pipelines/minimax_h3/pipeline_minimax_h3.py:1990 in _denoise
```

at steps ~31 and ~41 of 49, once in warmup and once in the timed pass. Same
subsystem (device->host readback inside `_denoise`), same intermittency, different
presentation. Whether the hang and the SIGBUS are one bug is unknown.

Those logs were in a session scratchpad that has since been deleted, so only the
transcript record remains — **keep any new SIGBUS log.**

## Diagnostics worth collecting on the next instance

Before killing it:

1. `tt-smi -s > ~/hang-ttsmi.json` — ARC/DDR/ETH status while wedged.
2. Python-level stacks of every thread:
   `py-spy dump --pid $P --locals` (or `gdb -p $P -batch -ex "thread apply all bt"`).
3. Which device is stuck: after recovery, note the device id in the
   `Device N init: failed to initialize FW!` message. Hang 1 reported **device 9**.
   If the same id recurs, suspect that specific ASIC/link rather than the model.
4. `dmesg -T | tail -100` for PCIe/AER/IOMMU events (needs privileges; was not
   available to me).
5. The exact `+N s after step 21` offset — see above.

## Open questions

- Deterministic per working point, or flaky? (**experiment 1 answers this**)
- Always ~step 22–23, or did the two samples coincide?
- Is device 9 always the wedged one?
- Does it occur with FSDP off, once OOM is avoided (e.g. any 5 s case looped)?
- Is it related to the `No fused MM/RS config for (M, K, N) = (4736, 3584, 5376)
  on 8-9 core grid` fallback that fires on every Wormhole denoise step?
