# debug-ltx-audio-galaxy

## Verdict — audio vocoder trace HANGS at end_trace_capture on 4x8 (host/dispatcher CCL-finalize deadlock)
**2026-06-10 04:05** · `tt-metal@9de36f1912f`

**Mechanism:** tt-triage via MCP watchdog callback (`TT_METAL_OPERATION_TIMEOUT_SECONDS=60` + venv-python tt-triage on dispatch-timeout). Isolated repro: `prof_vocoder_forward.py::test_forward_traced_correctness[bh_4x8]` (added 4x8 param — T-shard factor 8 axis1 + channel-TP factor 4 axis0; random weights, no checkpoint/transformer → fast). Commit `9de36f1912f`.

**Verdict:** `Vocoder.forward_traced` hangs at `ttnn.end_trace_capture` (`vocoder_ltx.py:388`). The warmup forward AND the capture-region forward both **complete**; only the capture **finalize** deadlocks. Error: `TIMEOUT: device timeout, potential hang detected, the device is unrecoverable` (`system_memory_manager.cpp:757`).

**Signals:**
- ALL triage checks **pass** (`dump_callstacks`, `dump_running_operations`, `check_eth_status`, `check_noc_status`) — no stuck kernel waypoint, no NoC counter mismatch, no in-flight op at trip.
- Only anomaly: every eth core, all devices, `unexpected watcher enable value: 0` (informational).
- Per `interpretation.md` §Silent-hang: green callstacks + never returns ⇒ **host/dispatcher-side deadlock** — `end_trace_capture` coordinating the captured CCL-over-eth graph (`partition_channel` + `_all_gather_t`) across all 32 chips.

**Scope:** Not a model-code bug — a trace-infra × CCL-on-Galaxy interaction. Matches #46441 closing vocoder trace for 2x4 (no benefit there); on 4x8 it hard-hangs.

**Next step (if pursuing trace):** rerun with `TT_METAL_WATCHER=5` to localize the dispatcher deadlock; likely a tt-metal trace/CCL-capture limitation to escalate, not a quick model fix. Recommend pivot to **Fix A (eager)** for the actual #46423 perf goal.

**Recovery:** device was "unrecoverable" → `mcp__tt_device_reset` ×1 (clean, 32 chips).

**Artifacts:** `~/.tt-buddy/triage/2026-06-10T040000-tracehang/{triage_summary.txt,triage_output.txt}`. See [[ltx-audio-galaxy]].

## Debug — traced 4x8 audio-decode capture wedged the mesh below the inspector
**2026-06-10 03:50** · `tt-metal@33c23dbab29`

**Mechanism:** tt-triage (live, venv python) — exhausted.

**Verdict:** `test_audio_decode_girl[bh_4x8sp1tp0]` with `LTX_TRACED=1` hung during traced audio-decode warmup. The wedge is *below* the Metal inspector layer (no inspector data), so per-core callstacks are unavailable. PID 72453 is **D-state** (uninterruptible), holding the 32-chip mesh; SIGKILL cannot reap it — only a device reset unblocks the driver call.

**Signals:**
- tt-triage `ValueError: Log directory /tmp/tt-metal/inspector does not exist. Metal runtime is not running` — runtime wedged without writing inspector logs.
- Prior identical wedge showed eth-core timeouts (27-25, 26-25) + "Device 0 init: failed to initialize FW".
- Eager 4x8 audio decode runs fine (1188.5 ms warm); only the **traced** path wedges → likely a CCL all-gather deadlock inside `ttnn.begin_trace_capture` on the 4x8 audio path.

**Next step:** `tt:run` recovery → reset. Sanctioned reset (`mcp__tt-device-mcp__tt_device_reset`) is **unavailable** — MCP client disconnected, daemon CLI has no reset subcommand. Blocked pending MCP client reconnect (`/mcp`) OR explicit user authorization for `tt-smi -glx_reset` (MCP queue empty + only the dead bare-metal proc on device, so the multi-agent risk the red-flag guards does not apply now).

**Artifacts:** `~/.tt-buddy/triage/2026-06-10T034700-hang1/triage_output.txt`

**Harness lesson:** `LTX_TRACED=1` on the girl test warms the FULL pipeline (transformer + VAE) — wrong/heavy harness for measuring audio-decode trace. Never SIGKILL a device-holding process (wedges the mesh → forces a reset). See [[ltx-audio-galaxy]].
