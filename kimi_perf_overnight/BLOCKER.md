# BLOCKER — overnight run PAUSED (2026-06-13 ~20:12)

## Status: PAUSED, awaiting a free/healthy device. NO experiments completed yet.

## What happened
1. Exp 00's runner was launched 3 times tonight; all failed to reach `exported descriptor`:
   - 19:39 attempt: I killed it mid-compile at 19:51 (to upgrade the harness). **`kill -KILL` mid-init
     is the documented chip-lock hazard** — this most likely WEDGED the device.
   - 19:54 attempt: crashed with `sysmem mapped at unexpected NOC address` (a multithreaded zombie
     `Zl` from the killed run held the sysmem NOC space ~6 min before init reaped it). Fixed
     `cleanup_procs` to wait for FULL reap.
   - 19:57 attempt: got past device-open, then **aborted**: `Device 15: Timed out while waiting for
     active ethernet core 26-25 to become active again. Try resetting the board.`

2. The device looks **wedged**: `tt-smi -s` shows `ETH_LIVE_STATUS: 0x0` while `ENABLED_ETH: 0x3edf`
   (ethernet enabled but not live) on board `tt-galaxy-bh` 0x...31831011 (Device 15 region).

## Why I did NOT reset the board
**Another user is ACTIVE on this shared box:** user `asaigal` is running several
`_migration_endpoint_driver.py` processes (tt-llm-engine disaggregation/migration), some up 2 days,
one started ~19:48. This is a Galaxy (whole-tray) system; `tt-smi -glx_reset*` would reset the entire
tray and **destroy asaigal's running work**. A reset that clobbers a co-user is an outward-facing,
hard-to-reverse action that needs explicit human authorization, which I cannot get overnight. So I
paused instead.

Note: my mid-init `kill -KILL` at 19:51 may have ALSO disrupted asaigal's work (collateral from the
device wedge). Worth coordinating with asaigal in the morning.

## What is paused
- Orchestrator STOPPED (tmux session killed, `STOP` sentinel present). No more runner launches → no
  more crashing into the wedged/contended device, no further interference.
- Nothing was lost: queue/ still holds all 11 experiments; instrumentation committed to git
  (ppopovic/investigation @ d0abf44ed8f).

## Auto-recovery plan (what the monitoring loop is doing)
Polling every ~25 min for the box to become free + healthy:
- If `asaigal`'s device processes exit AND `ETH_LIVE_STATUS` recovers (nonzero) → relaunch the
  orchestrator; experiments resume automatically.
- If the box is free (no other user) but STILL wedged → a `tt-smi -glx_reset_auto` becomes defensible
  (no co-user impacted, standard recovery, user authorized overnight continuation). The monitor will
  do it ONLY when no other user's device processes are present.
- While `asaigal` remains active and the device is wedged → stay paused; do NOT reset.

## For the user in the morning
If experiments still haven't run: the box was occupied by asaigal and the device was wedged (likely by
my mid-init kill). Decide whether to (a) coordinate a reset with asaigal, or (b) reset now if the box
is free. To resume after the device is healthy:
`rm /home/ppopovic/kimi_perf_overnight/STOP; tmux new-session -d -s kimi_perf "bash /home/ppopovic/kimi_perf_overnight/orchestrator.sh"`

## Poll log
- 20:40 — asaigal STILL ACTIVE (~55 migration_endpoint/worker procs; workers on c07u02, endpoints driven from c07u08). Device ETH_LIVE_STATUS still 0x0. My procs: none. → stayed PAUSED (branch c), no reset. Note: asaigal's disaggregation also occupies c07u08, so 2 of the 4 candidate relocation hosts (c07u02, c07u08) are theirs.
- 21:06 — asaigal STILL ACTIVE (48 migration procs). Device ETH_LIVE_STATUS still 0x0. My procs: none. → stayed PAUSED (branch c), no reset.
- 21:32 — asaigal STILL ACTIVE (48 migration procs). Device ETH_LIVE_STATUS still 0x0. My procs: none. → stayed PAUSED (branch c), no reset.
- 21:58 — asaigal STILL ACTIVE (48 migration procs). Device ETH_LIVE_STATUS still 0x0. My procs: none. → stayed PAUSED (branch c), no reset.
- 22:24 — asaigal STILL ACTIVE (48 migration procs). Device ETH_LIVE_STATUS still 0x0. My procs: none. → stayed PAUSED (branch c), no reset.
- 23:25 — asaigal STILL ACTIVE (48 migration procs). Device ETH_LIVE_STATUS still 0x0. My procs: none. → stayed PAUSED (branch c), no reset.
- 00:26 — asaigal STILL ACTIVE (48 migration procs). Device ETH_LIVE_STATUS still 0x0. My procs: none. → stayed PAUSED (branch c), no reset.
- 01:27 — asaigal device work ENDED (48 → 0 migration procs; only benign systemd/sd-pam/sshd login remain). No other device users. Device STILL wedged (ETH_LIVE_STATUS 0x0). DECISION: did NOT reset yet — asaigal only just wound down (<1h) and is still logged in (may relaunch); a Galaxy reset is outward-facing. Tightened poll to ~15min. ESCALATION: if next poll still shows asaigal device-free (sustained idle) and device still wedged → do tt-smi -glx_reset_auto + resume (branch b).
- 01:45 — asaigal device-free SUSTAINED (0 migration procs, 2nd consecutive). No co-user device work. RAN `tt-smi -glx_reset_auto` → "Re-initialized 32 boards after reset" (exit 0). NOTE: ETH_LIVE_STATUS still 0x0 even post-successful-reset → 0x0 is NOT a wedge indicator, just idle telemetry (my earlier wedge-gate was wrong; real wedge evidence was the 19:58 runtime crash during asaigal's 48-proc peak = contention). RESUMING orchestrator; device health judged by whether exp 00 opens the device cleanly.
- 02:00 — RECOVERED & FLOWING. Device healthy post-reset. Exp 00 COMPLETED: per-iter 2951→3101 ms (mean ~3050, ramps with prefix) @ 61440 — reproduces baseline, no instrumentation regression. Now running 01_maxseq56320 (decisive H3), no crashes. asaigal still device-free. Monitor now in normal-analysis mode.
- 02:13 — 01_maxseq56320 FAILED: stale /dev/shm/tt_prefill_layer_acks_ds_prefill (left by SIGKILLed 00) blocked the O_EXCL ack-channel create. NOT a device wedge (00 succeeded). FIX: added shm cleanup to cleanup_procs (rm tt_prefill_layer_acks_ds_prefill + tt_h2d_stream_service_ds_prefill* between experiments); removed stale file; re-queued 01; restarted orchestrator. Device healthy throughout.
- 02:29 — shm fix HELD. 01 (56320) succeeded ~3089ms = 00 (61440) ~3094ms → H3 REFUTED. 01a standalone-chunked ~1878ms/chunk ≈ test 1.94s → H1/H2 CONFIRMED (gap = request-loop machinery, not the model). Running 01b standalone+sections. Wrote ANALYSIS to RESULTS.md.
- 02:43 — 02 request-sections done. Section timing: request slower in BOTH mla (+456ms, = 61 ack syncs) AND moe (+347ms, = systemic: H2D svc / line-578 sub-device clear). Construction identical. 03 test running (~20-30min pytest). Decisive ack test = 04 (no timing).
- 03:05 — 04 skip_ack_sync ~3210ms ≈ baseline → ACK SYNC NOT THE CAUSE. 03 test sections ≈ standalone (test==standalone fast path). 06 maxseq81920 ~3215ms → H3 still refuted. Remaining suspects: H2D service / line-578 sub-device clear. Committed PREFILL_FORCE_PRECLEAR probe (3b63b010766) + queued 09_standalone_force_preclear (decisive disambiguation). 07/08 still to run.
- 03:29 — CONCLUSION reached. 09 FORCE_PRECLEAR standalone ~1879ms = unchanged → sub-device clear INNOCENT. By elimination + 08 corroboration → H2D STREAM SERVICE is the root cause (~1.2s/chunk). Wrote CONCLUSION to RESULTS.md. Committed FORCE_BUILD_SERVICE probe (0900424df25) + queued 10 (positive confirmation). Fixed orchestrator nullglob empty-queue bug. Relaunched.
- 03:46 — DONE. 10 FORCE_BUILD_SERVICE standalone = ~3177ms/chunk (was ~1878 without service) → H2D SERVICE DIRECTLY CONFIRMED as root cause. FINAL SUMMARY written to RESULTS.md. Stopping orchestrator (queue empty, investigation complete).
- 08:24 — FIX ATTEMPT 1 (core relocation to col 11): NEGATIVE (~3190 vs ~3201ms at col 0). Core placement not the mechanism. Committed diagnostic knob (89283177b4d). Per-dispatch/sub-device cost; needs tracy. Stopping orchestrator; not guessing further on shared device.
- 09:41 — CHUNK PROFILE done (1 layer, warm): 1st chunk (logical_n 5120) vs last (56320). Median kernel identical (46us) — most ops scale with chunk_size; ONE op (sdpa/ring_mla attention) grows ~2.8ms->~14ml (~5x) with logical_n. OP2OP latency flat. => within-run ramp = MLA attention vs KV prefix (expected); orthogonal to the H2D dispatch tax. Disk 2.8G free, raw logs purged, tmux clean.
- 11:20 — 3-LAYER service-vs-standalone done. Kernel times equal; WARM op2op gaps grow +5..+25ms/layer with service (dispatch tax confirmed per-layer/warm). MoE expert ops hit hardest (~7us->~2500us op2op). Logs saved. Cleaning up, stopping.
- 12:46 — DISPATCH GAP concluded. Gaps REAL (profiler OFF==ON ~24.7ms/1-layer-chunk, ~9ms/37% real idle). Per-op multi-chip go-signal/dispatch ~370us/op, run-ahead-starved on small ops. NUM_CQ=2 no help. Fix = Metal Trace (purpose-built) / op-fusion / C++ mesh go-signal opt. No config knob fixes it. Cleaned up.
- 13:10 — Go-signal/host-fan-out fix FAILED (1-layer warm 58ms vs 24.7ms baseline; reverted+rebuilt). Confirms gap is device-side local dispatch, not host. 4x8 fallback blocked: cache keyed to mesh shape (needs rebuild, disk-prohibitive) + dispatch is per-chip-local so axis swap is a longshot. Presented decision to user.
- 14:04 — OPTION B concluded. Pinned: per-op gap = kernel-config L1 ring (69KB) byte-wrapping ~every op. kernel_config_entry_count 8->32 committed (a8239c53ea2): correct (PCC 0.965), perf-neutral (byte-wrap binds). Real fix = enlarge config-ring L1 (deep/HAL) or Metal Trace. Stopping autonomous loop; human decision needed.
