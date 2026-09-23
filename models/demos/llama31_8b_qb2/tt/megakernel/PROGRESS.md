# Experiment 2 resume note

Updated 2026-09-23T16:50:59.339115+00:00. All40 final paired processes completed and passed strict exact checks. Measured source `e5fa79c28ab9eb2f924aaf7eb0c53e3af7b25aad` on `codex/llama31-qb2-megakernel`. This checkpoint contains final tables/report. Final preservation completed successfully. All40 command/log/allocation/result JSON files match durable copies by SHA256; every tensor artifact has matching nonzero size after rsync transfer verification. All six final profile archives separately passed SHA256 on durable storage. Verification timestamp: 2026-09-23T16:55:46.690533+00:00. Measurement/report/safety-guard checkpoint `07b300e23072073d2d2981379410a1a19a82cb32`; final documentation head is recorded in `/data/moconnor/llama-minlat-114624/PARENT_CHECKPOINT.md`. No device work remains.

Start SHA `f776a26ce77921cc84331cafb1434ba20c6ec46b`; original base `b8915544692d8f9feb2c890afbc2f22791560cd2`. Isolated checkout, no main update. First experiment preserved at f776, default options and EXPERIMENT1_PROGRESS.md. Read [EXPERIMENT2_REPORT.md](EXPERIMENT2_REPORT.md) and [experiment2-benchmark-summary.json](experiment2-benchmark-summary.json) for final comparisons, execution plan, limits, failed experiments and reproducibility. WORK_LOG.md is chronological evidence.

Median of process medians, warmed unprofiled host ms/token:

| Context / outputs | Native | First resident | General candidate | Bounded short variant |
|---|---:|---:|---:|---:|
| 128 / 32 | 7.633649 | 9.248212 | 7.542283 | 7.513495 |
| 2048 / 32 | 8.074194 | 9.674421 | 7.966866 | unqualified |
| 8192 / 32 | 8.681360 | 10.335836 | 8.620870 | unqualified |
| 128 / 256 | 7.679083 | 9.292730 | 7.579589 | 7.554150 |

Three Latin-balanced process blocks×five trials per primary case; two blocks×five trials for each labelled short variant. All general-candidate paired blocks beat their corresponding native block. Denominators31/255 decode steps; sampling and final history readback included. Exact teacher logits, every teacher step, all64 request-touched K/V tensors including padding, greedy outputs and repeats. Fixed prompt/batch one only. Both native and prototype failed the separate provisional HF PCC0.99 diagnostic; broad quality remains unqualified. Initial native versus historical speed difference is unresolved; final comparisons use this reservation's matched runs.

General candidate flags:

```text
--projection-reader pipelined --projection-buffers 3 --wide-subblocks
--bounded-layer-barrier --alias-projection-cbs --projection-placement dram
--scratch-init-once all --early-weight-blocks 2 --multicast-layer-barrier
--coalesce-input --projection-tile-height 16 --norm-tile-height 16
--head-placement select --custom-gu --norm-stats-face
--attention-placement head_priority --require-exact
```

One resident program includes embedding/all32 layers/final gather/norm/head. Eight shared O/GU/down workers, eight separate QKV, native32 attention/chunk256. The bounded short variant replaces `--attention-placement head_priority` with `--attention-workers 16`; only128-start32/256-output workloads qualify. Do not apply it to larger contexts. Native prefill and traced sampling remain separate.

`--single-layer-barrier` is now rejected before hardware setup: full2048 hung. Its source is retained for investigation. The guard was added after the measurement freeze and does not change selected/default configurations. Live triage saved before stopping only its child. Native `Semaphore<>` uses templated get_semaphore, bypassing the legacy function-like bank-address macro, consistent with captured attention reducer waits. No repaired protocol was hardware-qualified; other races remain unassessed. Final candidate retains64 global joins. Last reset recovery-head-priority-single passed health/fullconnectivity/ring15:24:01. No later recovery was needed in the paired suite.

Required customGU UNPACK/MATH tensix_sync calls must remain. They fixed timing-dependent first-subblock errors. Early own-buffer weight reads are refilled and charged every layer/token; no free cross-request residency. Exact current buffer geometry/partial-rounding boundaries matter. Alias lifetimes, inactive -1 warmup, page migration and replay tests pass. Raw trace callers must reserve_invocations; bounded limit2^26 collective phases =1,048,576 full invocations; legacy2^25 =524,288. Rebuild body/global semaphores at limit; never reset only the host count.

Memory: staticCB maximum991232B/head core,592384B/projection core; loopstate4096B/core,1.5MiB L1/core. Pinned views overlap allocator tensors; not a full peak. Final8192 audit covers60ELFs, maxtext BR12000/NC7784/TR014788/TR113652/TR211180B. Full8192 prefill/capture/decode fits selected flags; other combinations are not implied safe.

Separate full8192 worker Watcher passes exact checks on all4 chips; ETH instrumentation excluded for code-size capacity. Final128 main/O/GU/down and8192 main-v2 profiles each coverall4chips×3windows×35ops; required32-layer/8-reader markers complete, unmatched0. First8192/4000-op capture missing warmup records is excluded;16000-op retry valid. Six final raw archives (including failed capture) are compressed/tested/durable and SHA256-verified. Chip2 cross-core offset invalidates all-chip firmware spans. Same-core max model-only BRISC intervals~7.10–7.11ms128/~8.18–8.19ms8192, sampling excluded. Do not sum overlapping waits or call NoC payload DRAM-bus counters.

Local root `/home/moconnor/llama-minlat-114624`, durable `/data/moconnor/llama-minlat-114624`, old mirror `/data/moconnor/llama-megakernel-113796` read-only. Source/installed/loaded libtt_metal BuildID467a3c083d79401b9b46f83db7a46430ca790654 matches. Clang20.1.8/CMake4.0.2/SFPI7.80[956]/Python3.12.3/Torch2.11CPU/Transformers5.12.1; tar and tt_pybinds both installed. Builds/env/caches local. Source `artifacts/run-env.sh`; exact final commands/order in final-paired-plan.json. Offline summarizer: `python -m models.demos.llama31_8b_qb2.tests.summarize_experiment2 --artifacts /data/moconnor/llama-minlat-114624/artifacts --output /tmp/summary.json`.

Exclusive job114624/exactnodeqb2-120-p01t01 expires17:12:44UTC23September2026. This job grants no access afterward. Begin final preservation16:52:44; all tests/builds/copies/checkpoint finished17:02:44; launcher17:07:44. All device tests ended16:50:05UTC; final preservation started before16:52:44 and completed before17:02:44. No new optimization jobs. Stop only owned sidebands; no reservation extension/replacement/release, firmware change or reboot. Future hardware work requires a new handoff, live exclusive-allocation verification, adapted run_device.py guard, matching build and healthy four-chip mesh. No push/merge/PR/external message was performed; parent can push verified incremental bundle using existing authentication. Bundle requires f776.

Next research: hold native32-worker attention arithmetic fixed while isolating head/activation delivery placement; separately repair both semaphore APIs before testing single-boundary protocol. Independent gate/up groups and a broader prompt suite remain unqualified. Preserve these limitations with the result.
