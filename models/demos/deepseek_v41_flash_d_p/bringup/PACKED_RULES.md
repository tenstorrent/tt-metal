# Owner rules for every PB (packed prefill) task

Binding for every PB.* task of the galaxy_8x4-full ledger (moved here from each brief, 2026-11-20, to keep tasks.yaml under the repo's 500 KB limit).

OWNER RULES FOR EVERY B TASK (2026-11-19).
FAST ITERATION IS THE FIRST RULE: never load what you do not need; before any load write the one-line reason in BREADCRUMBS; iterate on the CPU, one layer, the characteristic layers [0, 1, 2, 3, 14, 20, 21, 24], then layers 0-20, and the full 40 only once (the gate does it).
Every device open leaks 10-27 GiB of host RAM and a model load costs 2-4 min: ALL device work (op tests, probes, layer tests) goes through the resident runner (python -m models.demos.common.bringup.resident status|submit --spec <spec> --cmd "<safe-runner command>"); never run scripts/run_safe_pytest.sh directly (it makes the runner step aside = a fresh open).
New test files: BATCH_SHARED = True (+ BATCH_ITEM_VARS for per-run env, BATCH_LAYERS for the layers it needs).
A/B switches are call-time (read per chunk, listed in bringup/hooks.py batch_call_settings) so both sides run in one load.
New kernels: bounded waits, every awaited semaphore signalled, NOC writes inside buffers, fabric payloads <= 6208 B, smallest shape first (two hangs on 2026-11-19).
Only matmuls stay within 110 cores; other ops may use 120.
Determinism (bit-identical repeats) outranks everything.
Do not break tt-d-gen compatibility (served caches, formats, addresses, acks, chunk header, the engine's device input); the model does NOT know the next chunk; no per-prompt precompute, no extra host round trips.
No fork regression / unit suites on the 8x4: run only the tests you add and the model-level checks.
Every Python-built op goes through ttnn.bringup.prepare_program / run_prepared.
Forks first (ttnn/ttnn/bringup/INDEX.md), then TTNN.
Do not change thresholds, goldens, frozen tests, tasks.yaml, state or results/.
Write per-segment, never per-test special cases.
Record what you did and measured in BREADCRUMBS.md (PPB.<n> section).
Plan and numbers: /data/dnijemcevic/bringup/handoff_20261119/BATCHING_PLAN.md (the owner's plan: design, oracle, ladder), /data/dnijemcevic/bringup/handoff_20261119/packed8k_projection.md (projected, not measured), /data/dnijemcevic/bringup/handoff_20261119/HANDOFF.md.
OWNER 2026-11-19 20:45-21:00 (binding): GOAL = packed prefill AT THE PERF TARGETS, not just working: agentic text, steady, serving-equivalent: 4 x 2k each over ~139k history >= 67k new tok/s (<= ~122 ms a 8192 chunk) and the mixed-history case (61k / 139k / 308k / 547k) >= 64k.
NO SHORTCUTS, mimic serving: every chunk each segment's history comes from ITS OWN slot's served cache in tt-d-gen's layout and is gathered / refilled for that chunk (no per-chip gathered table or any per-slot device state assumed to persist across chunks or to belong to the same request; slots are arbitrary and change every chunk; the model does not know the next chunk); device memory budgeted honestly (all segments' history buffers resident at once, checked against the DRAM budget at 1M histories); inputs as tt-d-gen delivers them (device ids, a per-segment descriptor in the chunk header; no per-prompt host precompute, no extra host round trips, no hardcoding to a trace); measured with real texts at real history positions, padded and empty lanes included.
Anything needing a shortcut is an OWNER DECISION: stop and write it in BREADCRUMBS.
PIPELINE AND HIDE across segments from the start: per-segment fixed costs (history gathers, engram lookups, launches, host prep) must not add up on the critical path: segment i+1's lookup / gather / prep runs while segment i computes (fused with or concurrent to attention, the indexer or the token-wise work, as P.87 fused the engram lookup with the history prefetch; or hidden behind the engram lookups as today).
Judge by the device profile and the exposed-host breakdown; report exposed vs hidden gather time per segment.

TT-D-GEN SHAPES (owner 2026-11-20 16:00, from the tt-d-gen source, /scratch/dnijemcevic/tt-d-gen = origin/main Nov 6): one fixed chunk_size (a multiple of 32), every chunk padded to it, only a request's last chunk partial; every start 32-aligned (prefix resumes at 64-token blocks; DSv4 recurrent: requests start at 0, chunk starts are multiples of chunk_size, except a last chunk clamped to max_seq_len - chunk_size); lengths 1..chunk_size (1-token segments ARE reachable and supported).
Glued packing: segment starts 32-aligned, splits only at 32-aligned positions (the continuation starts 32-aligned; <= 31 pad rows a split).
Do not build or test unaligned starts (31, 33, ...).
