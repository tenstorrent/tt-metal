# Experiment 2 resume note

Updated 2026-09-23 15:01 UTC. Optimization continues. Exact short-context screens beat the refreshed native median; final paired confirmation is pending. Branch `codex/llama31-qb2-megakernel`; source `d4717017` plus this documentation. Start `f776a26ce77921cc84331cafb1434ba20c6ec46b`; original base `b8915544692d8f9feb2c890afbc2f22791560cd2`. First experiment remains reproducible at f776 and in EXPERIMENT1_PROGRESS.md. See WORK_LOG.md for chronological commands, failures and results.

Exclusive Slurm 114624, exact host qb2-120-p01t01, expiry 17:12:44 UTC. Begin final preservation 16:52:44; finish tests/builds/copies/checkpoint 17:02:44; launcher stops 17:07:44. Never extend, replace or release. All hardware serial through `artifacts/run_device.py`: live allocation/host/deadline check, flock, source SHA/patch, unique logs, bounded timeout with live triage before own-child cleanup. Last recovery 14:37:08 after custom-down Watcher abort; reset, health, connectivity and ring passed. No current RECOVERY_REQUIRED. Watcher worker cores only (`TT_METAL_WATCHER_DISABLE_ETH=1` because ETH instrumentation does not fit). No subagents or external messages/push/PR.

Local root `/home/moconnor/llama-minlat-114624`; durable `/data/moconnor/llama-minlat-114624`; old mirror `/data/moconnor/llama-megakernel-113796` read-only. `source ../artifacts/run-env.sh`. Builds, venv and caches local. Clang 20.1.8 / CMake 4.0.2 / SFPI 7.80 / Python 3.12.3 / Torch 2.11 CPU / Transformers 5.12.1. Source, installed and loaded libtt_metal BuildID `467a3c083d79401b9b46f83db7a46430ca790654` match; both tar and tt_pybinds installed. Sandbox bwrap is unavailable; reviewed require_escalated calls work. Previous durable checkpoint `193d75ef`, ~14:06 UTC; this checkpoint is being preserved now. `commands/checkpoint.sh` creates verified incremental bundle plus notes/evidence; avoid copy I/O during headline timing. Parent can push the bundle with its own authentication.

Five warmed unprofiled trials, ms/token (128/32, 2048/32, 8192/32, 128/256):

| Path | 128 | 2048 | 8192 | Long |
|---|---:|---:|---:|---:|
| Native | 7.635744386 | 8.067759355 | 8.677945743 | 7.678513819 |
| First resident | 9.243840999 | 9.665413420 | 10.334084839 | 9.287472176 |
| Best all-case exact: custom GU | 7.656967936 | 8.084260776 | 8.751011514 | 7.699934576 |

Denominator 31/255 decode steps. Sampling and final history readback included. Every teacher logit, all 64 request-touched KV tensors, greedy output and repeat agree exactly. Fixed prompt/batch one, not broad quality. HF provisional 0.99 diagnostic failed for both native and prototype; do not claim otherwise. Native is ~1.15 ms faster than historical results for unresolved reasons; compare within this reservation. Final paired/interleaved comparison is still required.

Best all-case command flags:

```
--projection-reader pipelined --projection-buffers 3 --wide-subblocks
--bounded-layer-barrier --alias-projection-cbs --projection-placement dram
--scratch-init-once all --early-weight-blocks 2 --multicast-layer-barrier
--coalesce-input --projection-tile-height 16 --norm-tile-height 16
--head-placement select --custom-gu --require-exact
```

Results: `custom-gu128-result`, `custom-gu-qualification-{2048,8192,long}-result`. Eight shared O/GU/down workers near the eight DRAM banks, eight separate QKV workers, native 32 attention workers. No all-operation shared pool. Custom GU uses original raw [K,28] weights, 7-column Blackhole LoFi MVMUL, K-block 8, original BF16 L1 partials/final reload. A is compacted to 8-row format; partial/output stay 16-row to preserve face1 at DST row16. Explicit UNPACK and MATH tensix_sync before each custom call fixed timing-dependent first-subblock errors and are part of the qualified candidate. No DRAM repacking or extra weight payload. Default options preserve experiment 1.

Latest screens (strict exact, five unprofiled trials):

- `--norm-stats-face`: 7.641204291 at 128, ~15.76 us better than custom GU. Transfers 1024-byte first FP32 face instead of 4096-byte statistics tiles; all 16 row statistics retained. CB9/10 other faces zeroed first layer every invocation. Focused Watcher passes; other full workloads pending.
- `--cache-layer-table`: 7.657598227, no useful gain. Batches all 32 address rows per invocation into +4096 bytes/core. Full128 and focused Watcher exact; disabled.
- `--hoist-pack-config` on custom GU plus short statistics: 7.646269548, no useful gain; disabled. Focused Watcher exact. Initial compile failure after GU16 generalization was fixed by passing width as a template argument, rather than relying on GU_WORKERS in QKV/head includes. Clean device close, no reset.
- Current serial script `custom-gu16-components.sh` compares exact custom GU8/GU16 native layout/GU16 raw split-bank layout components, then `custom-inline128.sh` runs all-core inline CB reset Watcher and full128. GU16 Watcher already passes 12 stage checks/replays. Resident GU16 composition still rejected by Preparation/CB alias guards.

Earlier failures that constrain choices: shared QKV pool 8.626 vs 8.506 ms; head helper prefetch 10.589 vs 8.506; down helper prefetch >2 ms loss; DRAM-near norm-helper GU loses. Compact activations lose after 16-row geometry. Repaired fullDST exact128 8.0825 and custom QKV 8.0132 lose 7.9378 control. QKV8 buffers exceeded L1 by 24,448 bytes and required recovery. Head own-prefix 2 saved only ~6 us before custom GU and needs a new paired test. Inline reset previously reached only some custom kernels; the new all-core flag propagation needs requalification.

Attention count is not a general shortcut. 16 workers/chunk256 exact128=7.8085 before custom GU, but 8192 failed PCC .990715/relL2 .1361 and was slower. Eight workers exact128/long, but 2048 failed PCC .999092/relL2 .03472 and was slower. Chunk128 failed focused PCC .999892. Secondary gate was declared in advance: each teacher step and all KV PCC>=.9999, relL2<.01, exact top1/greedy/replay. No failed tolerance was relaxed. Short-context attention16 + custom GU remains a useful bounded test, not yet measured; head placement also changes with free cores.

Early weight reads refill existing rings every layer/token, before activation joins, then skip identical later reads; no uncharged residency. Qualified selective `profile-combined-{main,o,gu,down}` captures have all 4 chips, all 32 layers and 3 windows; median same-core completion lead QKV 10 us / O 61 / GU 21 / down 11. These are not additive latency. Chip2 has ~21.6 ms cross-core clock offset, invalidating aggregate firmware spans; do not report an all-chip device wall time. Archives are compressed, tested and durable with SHA256. Final candidate profiles still needed. Current source marker precheck: 80 locations, zero collisions against baseline zone log.

Memory: default 86 layer + 24 terminal workers; static CB maximum 991,232 B/head core, ~592,384 B/projection core. Aliases count once; pinned views overlap allocator tensors. Custom-GU8192 ELF max text BR12000/NC7784/TR0 14788/TR1 13652/TR2 11180 B; these are not complete L1/stack/firmware usage. Blackhole L1 1.5 MiB/core. Descriptor plans and allocator snapshots are outside timing. Counter API budget <=2^26 collective phases =1,048,576 full invocations in bounded mode; original <=2^25=524,288. Direct, warmup/capture and generator replay reserve this budget. Raw trace callers must reserve_invocations; rebuild body/global semaphores at limit, never reset count alone.


15:01 update: exact custom GU + short norm statistics + attention16 screens are 7.511882840 (128/32) and 7.554243353 (128/256), versus initial native 7.635744386/7.678513819. Flags are the all-case custom-GU flags above plus `--norm-stats-face --attention-workers 16`. These are fixed short-context results only. The larger context attention16 failures remain disqualifying. Paired confirmation pending.

Controlled terminal-placement test: attention16 with `--head-placement select32` holds the original head/norm cores and gives 7.613639873. Thus about 102 us of the 129 us gain comes from terminal placement, about 28 us from the reduced pool under fixed head placement. Head prefix2 on the short variant gives 7.515197808, no gain. Next promising placement hypothesis: retain all32 attention workers, place head/norm as in the short variant, and move the displaced attention workers onto its otherwise unused cores. This has not yet been implemented.

Other new losses: custom O exact128 7.678662581; zero-padded custom down exact128 7.647344936; read lookahead3 with3 buffers 8.243528518, with4 buffers 7.705036708. GU16 custom components 159.01 us/native bank rows,157.76 us/repacked half-bank rows versus GU8 144.73 us. All remain opt-in and disabled. Chunk512 attention passes the focused test but fails full8192 (PCC .998642683, relL2 .0505443) and is slower8.834057807; rejected with unchanged thresholds. Chunk1024 is exposed but untested.

`--single-layer-barrier` is a new exact128 screen at7.631038357 (~10 us improvement over short norm, only4.7 us below initialnative; paired confirmation pending). It maps local semaphores to alternating32-slot banks in unused state words640..895, initializes both each invocation, and resets only the bank from L-2. Late old-layer arrivals cannot hit the new bank. All local RISC barriers/ordinary CB resets remain. Next layer's global start joins prior local completions/resets; final global end is retained:33 global joins instead of64. The terminal norm trigger explicitly uses its original program semaphore. No new L1 allocation. Four distinct layers0/10/20/31 pass strict Watcher, unused pages, remapping, inactive warmup and replay. The first command named four-layer still ran two because the old fixture sliced(0,31); it is not four-layer evidence. The corrected `single-barrier-four-layer-watcher2` is valid. Source changes are in67c39286/d4717017. All-case qualification still pending.

Custom-down Watcher failure was an overlapping CB-view check: its larger down ring crossed the end of the inactive smaller GU alias. DMA now splits at that view end with all Watcher checks enabled; focused/full exact tests pass. Failure/abort logs preserved; post-abort triage lacked live RPC/serialized inspector and is not live triage evidence. Recovery14:37 passed all four chips.

Next: finish checkpoint copy before headline timing, test the32-worker head-priority placement hypothesis, qualify the strongest general/short candidates, and collect final separate Watcher/selective profiles. Start balanced native/first/candidate all-case measurements by16:00 latest (~35–45 minutes), preferably15:40. Final report needs paired trials, setup/memory/code, exactness limits and restart instructions. Mandatory within-two-hours checkpoint is due before15:12:44; this15:01 checkpoint fulfills it once durable copy completes. Final preservation begins16:52:44; all work ends17:02:44.
