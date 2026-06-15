# Session context — Kimi prefill perf overnight (updated 2026-06-13 ~20:12)

> **STATUS 03:46 (Jun 14): DONE / CONCLUDED. Orchestrator stopped.** Root cause of the ~1.2s/chunk
> runner-vs-test gap = **the H2D stream service** running in the request-loop process. CONFIRMED two
> ways: (1) by elimination — mla_seq_len/H3, ack-sync, construction, and the line-578 sub-device clear
> all REFUTED; (2) DIRECTLY — exp 10 built the UNUSED service in the fast standalone path and it slowed
> ~1878→~3177 ms/chunk, matching the request loop ~3090. Standalone/test ~1878ms. Fix is RUNNER-side
> (separate command queue / shrink H2D_SYNC_WORKER_CORES off the compute grid / passive event-driven
> sync); the model (forward_chunk) is already at test speed. Full story + experiment table + fix in
> RESULTS.md "FINAL SUMMARY". Probes on branch ppopovic/investigation (d0abf44ed8f, 3b63b010766,
> 0900424df25, c708cb07a4d). All 11 experiments completed; nothing left to run.

This file is a periodic snapshot so the work is resumable tomorrow on a fresh session. It is
re-written on each monitoring wake-up; check the timestamp above for freshness.

## What the user asked for
1. Investigate why the Kimi K2.6 chunked-prefill **runner** is ~3.3 s/chunk vs the **no-PCC transformer
   test** at ~1.94 s/chunk (a ~1.4 s constant additive per-chunk gap). Try: measure MLA duration test
   vs runner, check inputs/weights/construction are the same, remove synchronize_device in MLA, scan
   repo for more ideas.
2. **Run all night, survive this session dying** — so it runs in tmux, independent of Claude, with
   durable on-disk results and a recovery doc.
3. Scan the repo for new failure hypotheses (done — see HYPOTHESES.md).
4. Periodically save session context (this file).
Do NOT ask the user questions overnight — they are asleep.

## What is running right now
- tmux session **`kimi_perf`** runs `orchestrator.sh`: a queue-driven, no-Claude-in-the-loop runner that
  executes `queue/*.exp` one at a time (re-sorts each iteration so live additions reprioritize),
  appends to `RESULTS.md`, moves each finished `.exp` to `done/`. Idles when empty; `touch STOP` ends it.
- Mesh is shared 8x4; orchestrator serializes and force-cleans procs between experiments.

## Key files (all under /home/ppopovic/kimi_perf_overnight/ unless noted)
- `RESULTS.md` — durable findings, one section per experiment (THE output).
- `HYPOTHESES.md` — ranked hypotheses + experiment→hypothesis map.
- `RECOVERY.md` — how to attach/inspect/stop/add experiments.
- `orchestrator.sh`, `lib.sh` — the harness. `queue/`, `done/`, `logs/`.
- Instrumentation (committed to git branch `ppopovic/investigation`, commit `d0abf44ed8f`):
  `tt-metal/models/demos/deepseek_v3_d_p/tt/perf_probe.py` (new),
  `tt/tt_prefill_transformer.py`, `tt/tt_prefill_block.py`, `tt/mla/mla.py`.
- Prior writeup: `tt-metal/models/demos/deepseek_v3_d_p/tt/runners/RUNNER_PERF_INVESTIGATION.md`.

## Env-gated probes added (default OFF, zero overhead)
- `PREFILL_SECTION_TIMING=1` — per-chunk `[section-timing]` embed/mla/moe breakdown (sync-bracketed).
- `PREFILL_DUMP_CONSTRUCTION=1` — one-shot `[construction-dump]` + `[input-dump]` tensor specs.
- `PREFILL_SKIP_ACK_SYNC=1` — keep per-layer ack inject, drop its synchronize_device (mla.py).

## The leading theory (after repo sweep)
The gap is most likely **request-mode machinery**, not forward_chunk compute itself. Top suspect (H1):
the request-mode-only `clear_loaded_sub_device_manager()` at `prefill_runner.py:578` changes the
sub-device baseline that the per-layer MoE load/clear (~60×/chunk) and the shared CCL manager assume.
The single decisive experiment is `01a_standalone_chunked` (same pipeline, no request machinery). See
HYPOTHESES.md H1/H2. Backup theory H3 = mla_seq_len (61440 vs 56320), tested by the scaling sweep.

## Experiment queue (runs in this order; re-sorted live)
00 baseline · 01 maxseq56320 · 01a standalone_chunked(DECISIVE) · 01b standalone+sections ·
02 runner+sections · 03 test+sections · 04 skip_ack_sync · 05 maxseq56320+sections ·
06 maxseq81920 · 07 maxseq102400 · 08 disable_ack+sections.

## How to resume tomorrow
- Read `RESULTS.md` top-to-bottom; cross-reference HYPOTHESES.md.
- Per-iter runner ms come from `pipeline.prefill() =` lines; standalone gives total/11; test gives
  `[section-timing]` + pytest timing.
- If unresolved, the un-queued code-edit experiments are listed at the bottom of HYPOTHESES.md
  (skip line-578 clear; STRICT_INIT; finer MLA sub-section timing).
- To ask Claude to continue: point it at this folder + the investigation branch. Memory note
  `kimi-perf-overnight` (in the auto-memory) also indexes this.

## Decisions / rationale worth remembering
- Restarted the orchestrator once early (00 had no results yet) to add the `standalone` driver after the
  4-agent sweep showed it was the decisive control — worth the ~2 min.
- Instrumentation lives in `forward_chunk`/block/mla so BOTH runner and test hit identical probes →
  apples-to-apples. Sync-bracketing inflates absolute ms but the runner-vs-test delta is valid.
- Committed probes to git (not pushed) for durability; pure-Python, no C++ rebuild needed.
