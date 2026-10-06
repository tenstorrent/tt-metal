---
name: tt-lab-ops
description: Run device jobs safely on the shared Blackhole Galaxy hosts - host list and bad hosts, per-host flock wrapper, hang watchdog thresholds and triage, resets, build slots and NFS stagger, detached launches, one process per scenario, paired A/B measurement, tmux/session-restart safety. Use before launching, monitoring, killing or resetting any device job, and before quoting any timing.
---

# Lab operations (shared hosts, shared NFS, many agents)

Last validated against commit: <none yet>
Evidence: `bringup/GUIDE_dsv4_lessons.md`. Intended kit tools (branch `ssinghal/bringup-tools`, `models/common/bringup_kit/`; check they exist, else
use the DSV4 originals in `models/demos/blackhole/deepseek_v41_flash/tools/`): `launch.py` (locked, detached, watched launch), `host_health.py`,
`hangwatch.py`, `results.py` (parse logs into LOG entries), `status_report.py`.

## 0. Hard rules
1. One device job per host. All device runs go through the flock wrapper on `/tmp/dsv4_dev.lock` (pattern: `tools/devrun.sh`, `tools/mkwt.sh`);
   make a variant pointing `TT_METAL_HOME`/`PYTHONPATH` at YOUR worktree. `TT_METAL_CACHE=/mnt/tt-data/ssinghal/tt-metal-cache/h<host digits>`.
2. Never edit, check out, commit in, or rebuild the main tree `/mnt/tt-data/ssinghal/tests/tt-metal` (benchmarks run from it). Work in your own worktree on your own branch.
3. Never kill, reset or "clean up" a process or device you did not start. Not even a suspected duplicate: ask the coordinator/owner (an agent once killed a believed duplicate that held a build slot). Kill only your own pid; log it.
4. Check the host first: `pgrep -af python_env | grep -v grep`, `flock -n /tmp/dsv4_dev.lock true && echo idle`. Hosts: 10.82.97.{30..35,40..48} (ssh by last octet). Bad: .31 faulty (wrong first tokens in some mesh rows; discard its numbers), .42 hangs at the first device op (check `tt-smi`, triage before use). Update this list in LOG when it changes.
5. Every number you report carries commit + host + log path + shape (layers, batch, ISL) + flags (`tt-bringup-log`). Always run ALL layers for an end-to-end number (`DSV41_LAYERS`-style variable set explicitly, no leftover env vars from a previous shell: `env | grep DSV41`).

## 1. Worktree for a track
```
git -C /mnt/tt-data/ssinghal/tests/tt-metal worktree add /mnt/tt-data/ssinghal/wt/<WT> -b <BRANCH> <base>
# link ignored build artifacts like wt/vllm_dsv4: runtime, build, build_Release, python_env, .cpmcache, tt_metal/pre-compiled, ttnn/ttnn.egg-info -> main tree; give it its own generated/
```
Worktree creation on NFS takes minutes: run it in the background. Logs: `/mnt/tt-data/ssinghal/<model>-logs/<track>_<what>.log` (own prefix only).

## 2. Launching
* Detached (survives your shell): `nohup setsid <wrapper> "<cmd>" > <log> 2>&1 &`, then record pid, host, log, commit in LOG. Attach the watchdog to the PYTEST PID, not to the flock waiter (a watcher attached to a lock waiter caused two spurious device resets).
* One process per scenario class: spec-decode scenarios, different batch sizes and different ISLs run in separate processes unless the session driver
  supports `reconfigure` (then check DRAM/L1 is identical after reconfigure back; see `RECONFIGURE_NOTES.md`). Two ISLs in one process leaked DRAM until fixed: cheap two-ISL check first.
* Build before benchmark: a full build takes 25-90 min (NFS, ~640 GB read). Develop with 2-4 layers; do ONE long session per candidate with several scenarios.
* Build slots [L-slots]: `tt/build_slots.py` flock slots on NFS, `DSV41_BUILD_SLOTS` (default 5; the grid used 10), `DSV41_BUILD_STAGGER_S` minutes between build starts, builds with < 9 layers skip the cap. A silent log with "slots busy, waiting" is not a hang.
* Do not leave `sleep`/`until` shell loops behind; use Monitor-style waits and end them when the job ends.

## 3. Hang watchdog and triage
* Thresholds by phase (the 8-10 min default killed healthy builds: weight load + Engram-like host table load are silent for ~9-15 min): build/load phase >= 45 min silent;
  compile/trace-capture >= 20 min; timed loop >= 15 min. State the phase in the launch.
* On a hang: `python tools/tt-triage.py --skip-version-check --disable-progress --disable-colors --llm-output-path=<report>` while the process is alive (needs the live runtime), save to `<logs>/triage/hang_<host>_<pid>_<hhmm>.txt`,
  kill your pid, then `tt-smi -glx_reset` under the device lock (only on your own host, only if nobody else holds the lock). Known signature: main thread in `FDMeshCommandQueue::wait_for_outstanding_reads` at the first op.
  After a reset run a 1-minute smoke test before long jobs. A killed hung job leaves the device dirty: the next job hangs on its first op until reset.
* A hang is normal: detect, triage-save, kill own pid, reset, retry once on another host. Second failure with the same signature: stop, log, report. Do not retry blindly in a loop.
* `import` of some demo modules opens the device driver: do not import them on a busy host (`Read 0xffffffff over PCIe`).

## 4. Measuring
* The first call in a session is slow (compile, readback ops) [L-first]: warm up, report steady-state steps (>= 5), and report step 0 separately.
* Host-to-host speed differs up to ~10%: only paired comparisons INSIDE ONE PROCESS (A/B env switched per scenario, e.g. a scenario-list env + session driver) count as gain claims. Cross-host/commit comparisons are hints.
* Decode ms/token depends on session type (dedicated single-scenario vs multi-scenario session), DSV41_MEMLOG state and commit; the size and even the sign of the difference varied by batch and its cause is NOT isolated [L-decms, inferred/open]. Never quote a decode ms/token without stating session type, MEMLOG state and commit; for a gain claim use a paired in-process A/B.
* Random-token MoE profiles are unreliable in either direction (two agents saw opposite signs); always use real prompt tokens and the full layer set [L-skew].
* Profiler distorts ms/token: use it only for op tables, not for throughput.

## 5. Session restart / machine safety
Run the agent session inside tmux/screen on a login host; a restart kills agents but not detached jobs. After a restart: list processes per host (`pgrep -af "pytest|tt-triage"`), reattach watchdogs to surviving pids, read logs, write a `handoff` LOG entry. Jobs whose agent is gone are still owned by the lab: do not kill them until the coordinator decides.

## Escalate / stop
Ask the human before: resetting a host with unknown holders, installing packages into the shared python_env, C++ rebuilds of shared libs (use a private build dir and ask), deleting weight caches/dumps/logs of others, anything beyond your assigned hosts. Stop and report if no idle host after 3 h, or two hangs with the same signature.
