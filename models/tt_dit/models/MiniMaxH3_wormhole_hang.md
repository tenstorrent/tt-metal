# MiniMax-H3 t2va on Wormhole 4x8: the intermittent mid-denoise hang

Observed 2026-09-16 sweeping `test_t2va_end_to_end` on a Wormhole galaxy (4x8, 32 chips):
2 hangs in 12 completed long runs, blocking the last 5 of 18 working points. This document
records what the two logs proved, the cause that follows from it, and the fix.

Raw logs on the run host: `~/h3_wormhole_results/sweep_fsdp.log.gz` (hang 1),
`sweep15b.log.gz` (hang 2).

## Where the hangs actually were

The earlier revision of this file placed the stall at "~step 22-23" by dividing elapsed time
by the per-step rate, and treated the two hangs' near-identical +13 s offset after `step 21/49`
as the most discriminating number available. Both of those readings were wrong, and the log
contains enough to do much better.

The `No fused MM/RS config ...` warning fired **once per transformer layer, 50 per denoise
step**, so counting warnings after the last `step 21/49` line locates the stall to the layer:

| | hang 1 (`9x16_10s`) | hang 2 (`16x9_15s`) |
|---|---|---|
| warnings after `step 21/49` | 150 = exactly 3 steps | 100 = exactly 2 steps |
| per-step dispatch burst | 50 warnings in 0.30 s (~6 ms apart) | 50 in 0.30 s |
| per-step blocking gap | 6.28 s, then 6.23 s | 12.35 s |
| **stalled in step** | **i=23 (`step 24/49`)** | **i=22 (`step 23/49`)** |

Two things follow immediately.

**The +13 s agreement is arithmetic, not evidence.** The hangs are at *different* step counts
past the log line — 3 steps at 6.3 s vs 2 steps at 12.4 s — and `2 x 6.3 ~= 1 x 12.4`. The old
argument ("if the trigger were step-aligned the offsets would differ") does not hold: with
per-step rates in a 2:1 ratio, a step-aligned trigger produces exactly this coincidence. The
number discriminates nothing.

**Every step has the same shape: ~0.30 s of host dispatch, then one long block.** The host
enqueues all 50 blocks asynchronously in 0.3 s and then blocks for the entire step time. The
only blocking call in the loop is the readback at `pipeline_minimax_h3.py:2002`
(`local_device_to_torch(video_velocity)`), so host run-ahead is bounded to one step, and a hang
means the device stopped retiring the step whose blocks were just dispatched.

That is also **the exact line of the two earlier SIGBUS crashes** (`utils/tensor.py:349`, from
`_denoise`), which were recorded here as "possibly the same underlying bug". They are the same
line and the same subsystem; treat them as one bug until something separates them.

## What is not the cause

- **Nothing in the loop is step-dependent.** The one shape change — the distinct-noise-level
  count feeding `tt_timestep` — happens only between i=0 and i=1. Computing both schedules
  (video shift 12.0, audio shift 3.0) shows they collide only at t=0 and are distinct for every
  i>=1. Steps 1-48 are an identical op sequence, so this is state drift, not control flow.
- Not memory, not canvas, not duration alone, not accumulated load, not the weight cache --
  all as previously established.

## Cause: Wormhole was taking the fused MM/RS path by accident

Those 50 warnings per step were not noise; they were the bug announcing itself.

`transformer_block_minimax_h3.py` gated the fused ff2 matmul+reduce-scatter on
`has_mmrs_config(m, k, n)`, which checked only `k == 3584 and n == 5376 and m % 32 == 0`. That
gate is architecture-blind, but *both* ways of resolving a real blocking are not:

- `_SWEPT_BLOCKINGS` registers under `_DEVICE_GRID = CoreCoord(12, 10)` — a Blackhole grid.
- the v2.3 rule engine in `get_fused_mmrs_config` is `is_blackhole()`-gated.

On Wormhole's 8x9 grid both miss, so every ff2 fell through to `default_fused_mmrs_config` —
precisely the case the gate's own comment exists to prevent:

> Only take it for a shape with a swept blocking. The generic fallback config runs the matmul
> on 56 of the device's 120 cores at subblock 1x1, making the fused op a measured 45%
> regression on this stage.

So the code documented an invariant it did not hold on this architecture, 50 times per step.

What that fallback actually configures is worse than slow. `FusedMMRSConfig.get_params` derives
the reduce-scatter worker count as `rs_zone_capacity // (2 * num_links) - 1`, which on the
Wormhole default (`CoreCoord(8, 7)` matmul grid, 8x9 device, `num_links=4`) is
`((9-7)*8) // 8 - 1` = **1 worker per link** — with `mm_window_blocks=2`, so `use_l1_handoff`
is true and the op runs its credit-based MM<->RS flow control. There is no floor or assert on
that expression; at `num_links=8` it yields 0. A credit-starved, one-worker-per-link fused
collective, on a blocking never swept for this architecture, running 50x per denoise step, is
the best available explanation for an intermittent stall in exactly this op.

It also fits the evidence better than DiT FSDP does. FSDP is what *changed* when the hang
appeared, but the fused ff2 path fires with FSDP on **and** off — which matches the SIGBUS
(seen FSDP-off) and the hang (seen FSDP-on) being one bug. FSDP's real contribution was
removing the OOM ceiling so long runs could reach these steps at all.

## The fix

Gate the fused path on whether a measured or rule-derived blocking **actually resolves for the
real device core grid**, instead of on the shape alone:

- `utils/matmul.py`: `get_fused_mmrs_config`'s precedence-1 and -2 resolution is factored into
  `_resolve_fused_mmrs_config`, and `resolves_fused_mmrs_config(M, K, N, grid)` answers the
  same question as a pure predicate — so the gate can never drift from the resolver.
- `mmrs_config.py`: `has_mmrs_config` / `register_mmrs_config` take the device core grid.
- `transformer_block_minimax_h3.py`: passes `mesh_device.compute_with_storage_grid_size()`.
- `FusedMMRSConfig.get_params` now rejects a derived `num_workers_per_link < 1` instead of
  passing it to the op. The expression is unbounded below and the op takes an explicit count,
  not an auto sentinel, so a zero would have configured a collective with no drainers and
  deadlocked. Unreachable at the link counts tt_dit actually uses (1, 2, 4) and at every
  blocking in `fused_mmrs_configs` — it is a tripwire, not a behaviour change.

Blackhole is unchanged by construction — the predicate still returns True there via the swept
table or the rule engine, verified across the H3 ff2 Ms. Wormhole now takes the ordinary
`matmul` + `reduce_scatter_minimal_async` path, and the warning stops.

Measured side effect on `1x1_5s` (M=2752), the shape with the cheapest step:
**1477 -> 1408 ms/fwd, 4.7% faster**, CLIP 36.32 vs 36.33 baseline. The fused fallback was
costing time, exactly as its own comment predicted.

## Symptom fingerprint (if it recurs)

- No further log output, indefinitely. No exception, no traceback.
- Process **spinning, not idle**: 171-316% CPU, CPU-time climbing past elapsed
  (measured 06:23:27 CPU over 02:01:14 elapsed). Every thread in `futex_wait_queue`:
  ```bash
  P=$(pgrep -f "^python -m pytest models/tt_dit" | head -1)
  ps -o pid,stat,etime,time,wchan:24,pcpu -p $P
  for t in /proc/$P/task/*; do echo "$(basename $t) $(cat $t/wchan)"; done | sort | uniq -c
  ```
  That is host dispatch busy-polling a device that stopped retiring work.
- **`@pytest.mark.timeout` does not fire** — pytest-timeout cannot interrupt a C-level stall.
- Afterwards the board is wedged. Both of these were seen:
  ```
  Device 9 init: failed to initialize FW! Try resetting the board.
  RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID: ..., ETH core e9-0
  ```
  A graceful `kill -TERM` exits cleanly but does **not** un-wedge the board.

## Running it so a hang leaves evidence

The old recipe (shell `timeout`, then reset) destroyed the evidence it needed. Turn the stall
into a raised timeout with an automatic device-state dump instead — the mechanism CI uses:

```bash
export TT_METAL_HOME=/home/jameslee/tt-metal
export TT_METAL_INSPECTOR=1
export TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1
export TT_METAL_OPERATION_TIMEOUT_SECONDS=300     # >> the 12.7 s worst-case step
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="$TT_METAL_HOME/tools/tt-triage.py --disable-progress --triage-summary-path=$OUT/triage_summary.txt --sqlite-output-path=$OUT/triage.sqlite"
```

Add `TT_DIT_LOG_EVERY_STEP=1` (added with this fix) so the step index is read off the log
rather than reconstructed, and so `step N dispatched, reading back` separates "dispatch never
returned" from "readback blocked" — previously indistinguishable, since the readback is the
only blocking call in the loop.

Watcher (`TT_METAL_WATCHER=30 TT_METAL_WATCHER_APPEND=1`) gives per-RISC waypoints and NOC
sanitization if the triage dump is not enough; it perturbs timing, so hold it in reserve for a
race this sensitive. The triage scripts that matter here: `dump_op_mesh` (op-ID skew across the
4x8 mesh — shows which chip stopped), `dump_callstacks`, `check_eth_status`, `check_noc_status`.

Recovery, after confirming nobody else is on the box:

```bash
for p in $(ls /proc | grep -E '^[0-9]+$'); do
  ls -l /proc/$p/fd 2>/dev/null | grep -q tenstorrent && \
    echo "pid $p user=$(stat -c %U /proc/$p) cmd=$(tr '\0' ' ' </proc/$p/cmdline | cut -c1-60)"
done
tt-smi -r      # tt-smi is 5.2.0 here; -r supersedes the deprecated -glx_reset
python -c "import ttnn; d=ttnn.open_mesh_device(ttnn.MeshShape(1,1)); print('OK'); ttnn.close_mesh_device(d)"
```

## Exposure, if a reproducer is still needed

Denoise steps actually executed across both sweeps, by per-device ff2 M:

| M (rows/device) | steps executed | hangs | s/step |
|---|---|---|---|
| <= 7040 (5 s cases + 4x3/1x1/3x4 10 s) | 882 | 0 | 1.5 - 4.8 |
| 9184 (10 s wide cases) | 269 | 1 | 6.6 |
| 13664 (15 s cases) | 121 | 1 | 12.7 |

~195 steps per hang at M >= 9184 and none below, which is suggestive but not significant at
n=2 (Fisher p ~ 0.09). It does mean the 10 s / M=9184 cases are the best instrument: they are
in the implicated group *and* give the most steps per hour (~545/h vs ~283/h at 15 s). A soak
on `16x9_10s` should surface a hang within roughly 20-40 minutes if the old rate still holds.

## Open questions

- Does the hang survive the fix? **No recurrence.** 980 denoise steps across 10 standalone
  runs, then a full 18/18 sweep (1764 more steps, 3 h 14 m, zero failures) — 2744 steps total,
  no hang, no SIGBUS, no MM/RS warning. Under the pre-fix rate (2 hangs / 390 steps at
  M >= 9184) that is p ~ 2e-04 -- computed over the **1666** of those steps that were at
  M >= 9184, since that is the only range the pre-fix rate was ever measured in; pooling all
  2744 would overstate it. Both cases that originally hung
  (`9x16_10s`, `16x9_15s`) now pass, in the sweep and standalone.
  **The honest caveat:** the pre-fix rate is estimated from only 2 events, so its confidence
  interval is wide and this is strong evidence rather than proof of a negative. What raises it
  above a bare p-value is that the mechanism was independently confirmed (the fused op really
  was configured with 1 RS worker per link on this grid) and removing it made the model
  *faster* -- so the change removed a real defect, not just perturbed the timing.

- If it ever recurs, the next suspect: FSDP's ~200-250 extra SP-axis ring all-gathers per step
  rotate a **2-deep**
  ping-pong semaphore pool (`parallel/manager.py:311-314`) with `barrier_semaphore=None` on the
  persistent path (`manager.py:872`), sharing the SP axis with the ring-SDPA K/V gathers.
  `manager.py:53-56` already documents a related desync-or-hang hazard. That is suspect #2.
- Is device 9 always the wedged one? `dump_op_mesh` now answers this directly.
- Are the hang and the SIGBUS one bug? They are the same line; the `local_device_to_torch`
  readback lands in mmap'd hugepage sysmem, so a SIGBUS there is what faulting on a mapping
  whose device went away looks like. `dmesg -T | tail -100` on the next instance (needs
  privileges, not available to me) would show a PCIe/AER event if so — which would make this a
  platform bug, not a model one. **Keep any new SIGBUS log.**
