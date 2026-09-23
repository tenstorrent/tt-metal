# Experiment 2 resume note

Updated 2026-09-23 15:50 UTC. Final all-case candidate, separate full8192 worker Watcher and selective profiles are qualified. Final paired suite is next, after this checkpoint copy. Branch `codex/llama31-qb2-megakernel`, implementation source `fa1135fb`; start `f776a26ce77921cc84331cafb1434ba20c6ec46b`, original base `b8915544692d8f9feb2c890afbc2f22791560cd2`. First experiment is preserved at f776 and in EXPERIMENT1_PROGRESS.md. Full command/result/failure history is in WORK_LOG.md and external artifacts.

Exclusive job114624 on qb2-120-p01t01 expires17:12:44 UTC. Begin final preservation16:52:44; all tests/builds/copies/checkpoint finished17:02:44; launcher stops17:07:44. Never extend/replace/release. Hardware runs serially through `artifacts/run_device.py`, which checks live allocation/host/deadline, flock, source SHA/patch, and saves bounded timeout/triage evidence. No subagents or external messages/push/PR.

Local root `/home/moconnor/llama-minlat-114624`; durable `/data/moconnor/llama-minlat-114624`; old mirror `/data/moconnor/llama-megakernel-113796` read-only. `source ../artifacts/run-env.sh`. Builds/venv/caches local. Clang20.1.8, CMake4.0.2, SFPI7.80[956], Python3.12.3, Torch2.11CPU, Transformers5.12.1. Source/installed/loaded libtt_metal BuildID467a3c083d79401b9b46f83db7a46430ca790654 match; both tar and tt_pybinds installed. Sandbox bwrap unavailable; reviewed require_escalated calls work. Durable checkpoint8781559b and its artifacts completed15:03, satisfying the within-two-hours preservation deadline. Refresh it after final instruments and again at completion.

Five warmed unprofiled screening trials, ms/token:

| Path | 128/32 | 2048/32 | 8192/32 | 128/256 |
|---|---:|---:|---:|---:|
| Refreshed native | 7.635744386 | 8.067759355 | 8.677945743 | 7.678513819 |
| First resident | 9.243840999 | 9.665413420 | 10.334084839 | 9.287472176 |
| Final candidate | 7.544432613 | 7.995079643 | 8.617639774 | 7.578982313 |
| Short-context variant | 7.511882840 | unqualified | failed earlier geometry tests | 7.554243353 |

Final candidate: eight bank-near shared O/GU/down workers, separate eight QKV workers, native32 attention workers/chunk256, terminal placement selected before relocating the other16 attention workers. One resident program includes embedding/all32 layers/final gather/norm/head; native prefill and sampled trace boundary retained. Required flags:

```
--projection-reader pipelined --projection-buffers 3 --wide-subblocks
--bounded-layer-barrier --alias-projection-cbs --projection-placement dram
--scratch-init-once all --early-weight-blocks 2 --multicast-layer-barrier
--coalesce-input --projection-tile-height 16 --norm-tile-height 16
--head-placement select --custom-gu --norm-stats-face
--attention-placement head_priority --require-exact
```

All teacher logits, all64 request-touched KV tensors including padding, greedy outputs and repeats match exactly in all four cases. Denominator31/255 decode steps; sampling and final history readback included. Fixed prompt/batch one, not broad quality. Both native and prototype failed the separate provisional HF PCC0.99 diagnostic. Native is~1.15ms faster than historical results for unresolved reasons; only this reservation's matched pairs count.

Short variant replaces `--attention-placement head_priority` with `--attention-workers 16`; only128-start32/256-output workloads qualify exactly. Reduced attention pools failed long-context numerical gates and are not general alternatives. Controlled fixed-head test attributes~102us of the~129us short-variant gain to terminal placement and~28us to attention-pool change. The32-worker final candidate pursues that placement gain without changing attention reduction geometry. Full-pool head_priority_all is worse128 (7.62934) and slightly worse8192 (8.65431); keep constrained placement.

The customGU uses original BFP4 raw[K,28] weights, seven-column LoFi MVMUL, Kblock8, original BF16 L1 partial rounding/final reload. A uses compact8-row format, Partial/Out16 retains correct DST face1 location. Explicit UNPACK and MATH tensix_sync are REQUIRED: they fixed first-subblock timing-dependent errors. Early own-weight reads refill existing three-slot rings every layer/token before activation joins and skip identical later reads; no extra weight bytes or uncharged residency. Short norm statistics transfer one1024B FP32 face with all16 row statistics, zeroing remaining faces every invocation.

**Do not enable single-layer-barrier.** Combined head_priority/single-barrier passed128 and four-layer Watcher but hung in full2048. Live triage on all4 devices is preserved in final-candidate-qualification-2048-early-triage; initiating race unresolved. Manually stopped only child group99064 after448.9s. Recovery-head-priority-single reset, health/fullconnectivity and ring passed15:24:01. Placement-only2048 then passed; no current RECOVERY_REQUIRED. All final commands retain64 global joins. Worker Watcher excludes ETH because instrumented ETH code exceeds its config capacity; do not claim ETH Watcher coverage.

Other rejected directions: direct sharedQKV loses~120us in its controlled composition; terminal helper prefetch10.589vs8.506ms, norm-helper down prefetch>2ms loss; GU16 components slower thanGU8; deeper lookahead/compact activations/customO/customdown/fullDST/customQKV lost. Custom-down Watcher overlapping-CB alias failure fixed by splitting DMA at inactive alias boundary; all bytes/checks retained, exact but slower. QKV8 buffers exceeded L1 by24448B and required recovery. All are opt-in experimental modes, not selected defaults. Details and sourceSHAs in WORK_LOG.

Memory: max staticCB991232B/head core and592384B/projection core, 1.5MiB L1/core. Pinned views overlap allocator tensors; these are not full execution peaks. Final head_priority8192 audit: maxtext BR12000/NC7784/TR014788/TR113652/TR211180B. Counter API limit boundedmode2^26collective phases=1,048,576full invocations; originalmode2^25=524,288. Warmup/capture/direct/generatorreplay reserve; raw trace callers must reserve_invocations. Rebuild body+global_sems at budget, never reset counter alone.

Next commands: `commands/profile-final-selective.sh` (128main/o/gu/down,8192main, all4chips×3windows), `commands/final-watcher.sh` (full8192 exact), archive/audit/checkpoint. Then `commands/run-final-paired.py` reads final-candidate-flags.json: three Latin-balanced native/first/candidate process blocks percase, five warmed trials each, plus two process blocks for the separately labelled short variant for128/long (40 processes total). Target start~15:51, finish~16:46. No heavy copy/compression during headline runs. Summarizer `tests/summarize_experiment2.py` checks every run's identity/exactness/boundary and emits trials, process medians, paired deltas and setup costs. Report/README/final JSON still required.

Old valid profile-combined main/o/gu/down all4chips×32layers×3windows proves own-prefix reads complete before activation joins; same-core median leadsQKV10/O61/GU21/down11us are not additive latency. Chip2 cross-core clock offset invalidated aggregate firmware wall spans. Final selected profiles must be independently audited; never sum overlapping worker waits/firmware durations. Profile archives are compressed, tested and durable with SHA256.

15:50 instrumentation update: final-head-priority-watcher8192 passes all exact checks with worker Watcher on all4 chips; no ETH coverage. Final128 main/O/GU/down and8192 main-v2 all cover3 windows×4 chips×35 ops, required32-layer/8-reader zones complete and no unmatched markers.8192 first4000-op report failed missing warmup records;16000-op retry succeeds. Both captures preserved, onlyv2valid. Model-only max same-core BRISC firmware interval median ranges7.101–7.113ms at128 and8.180–8.190ms at8192; chip2 cross-core offset~22.25s invalidates all-chip firmware spans. Final profile archives compressed/tested/copied, SHA256manifest ready for checkpointverification. No heavycopy/compression while pairedsuite runs. Report draft is EXPERIMENT2_REPORT.md; final paired JSON/report still pending.
