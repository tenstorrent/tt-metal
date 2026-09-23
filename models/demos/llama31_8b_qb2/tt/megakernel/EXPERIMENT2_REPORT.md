# Experiment 2: resident QB2 decode

Completed on 2026-09-23. The selected resident program improves all four measured workloads against the native traced model in the paired comparison below. Measured source: `e5fa79c28ab9eb2f924aaf7eb0c53e3af7b25aad`. Exact agreement and separate instrumentation checks passed within the scope described here.

This experiment uses one exclusive four-chip Blackhole QB2, batch one, snapshot `0e9e39f249a16976918f6564b8830bc894c89659`. It continues `f776a26ce77921cc84331cafb1434ba20c6ec46b` in an isolated checkout without updating main. The first experiment remains reproducible at that SHA and through the default `decode_token` options. Native prefill and device sampling remain separate. Both native decode and the resident implementation use trace replay.

## Final comparison

| Context / outputs | Native traced | First resident | Selected resident | Paired latency reduction |
|---|---:|---:|---:|---:|
| 128 / 32 | 7.633649 | 9.248212 | 7.542283 | 1.202% |
| 2048 / 32 | 8.074194 | 9.674421 | 7.966866 | 1.306% |
| 8192 / 32 | 8.681360 | 10.335836 | 8.620870 | 0.697% |
| 128 / 256 | 7.679083 | 9.292730 | 7.579589 | 1.295% |

Times are ms/token. Paired reduction is the median of within-round percentage reductions; it need not equal the ratio of the two displayed aggregate medians.

| Context / outputs | Native process-median range | Candidate process-median range | Paired saving range, µs/token |
|---|---:|---:|---:|
| 128 / 32 | 7.632691–7.637361 | 7.540595–7.545525 | 91.37–92.10 |
| 2048 / 32 | 8.072304–8.075223 | 7.964196–7.969903 | 105.32–110.00 |
| 8192 / 32 | 8.680502–8.701274 | 8.619647–8.620977 | 59.52–81.63 |
| 128 / 256 | 7.678647–7.679129 | 7.579437–7.579644 | 99.21–99.54 |

The independently labelled short-context variant measured:

- 128/32: **7.513495 ms/token**, 1.568% paired reduction; process medians 7.511156–7.515833, individual trials 7.509670–7.517865.
- 128/256: **7.554150 ms/token**, 1.624% paired reduction; process medians 7.553976–7.554324, individual trials 7.553835–7.554528.

Every completed paired process passed the exact teacher/KV/greedy gate. Repeated generations within each process were identical. See [experiment2-benchmark-summary.json](experiment2-benchmark-summary.json) for all individual trials, process medians, commands, hashes and startup costs.

Startup measurements from these processes (kernel/model caches already populated) are separate from the score:

| Path | Model load median (range), s | Enable/synchronize median, s | Warmup/capture/full-generation median (range), s |
|---|---:|---:|---:|
| native | 71.207 (67.522–74.417) | 0.013149 | 0.781 (0.468–2.205) |
| first | 70.335 (65.720–73.558) | 0.015447 | 1.038 (0.497–3.144) |
| candidate | 71.225 (68.791–73.891) | 0.015623 | 0.756 (0.444–2.154) |
| short | 70.296 (68.595–71.762) | 0.016018 | 2.128 (0.436–2.837) |

The warmup aggregate includes prefill, any needed compile, trace capture and one generation; it is not an isolated trace-capture measurement. Per-case details and warmed request setup/TTFT values remain in the JSON. No cold-start speedup is claimed.

The score is warmed host-observed decode time, including sampled trace replay and final token-history readback. Prefill supplies the first output: 32 outputs contain 31 timed decode steps; 256 outputs contain 255. Model loading, initial compilation/capture, request setup and prefill are reported separately. The harness retains its original deferred readback boundary; neither the candidate nor its comparison omits sampling or KV work.

The final suite alternates native, first resident and candidate order across three process blocks for every case. Each process measures five warmed generations. The table uses the median of the three process medians. The JSON retains every trial, process median, paired within-round difference, source SHA, exact command and result hash. Within-process trials are not claimed as independent process-level replications. The short-context variant uses two process blocks (ten trials) for each qualified workload, is labelled separately, and is never substituted for the general candidate at longer contexts.

Initial refreshed screening medians were native 7.635744/8.067759/8.677946/7.678514 ms/token and first resident 9.243841/9.665413/10.334085/9.287472. Native was approximately 1.15 ms faster than the historical reservation; the cause is unresolved. The old 8.788292/9.220867/9.834495/8.829523 native values are historical evidence, not this experiment's denominator.

## Execution plan

One resident mesh program performs embedding, all 32 layers, final gather/norm/head. The same four chips tensor-parallelize every layer. It is a single-token resident body replayed through a trace, not a persistent multi-token serving loop or a chip-per-layer pipeline.

Each chip uses eight bank-near projection workers for O, packed gate/up and down at their dependent phases, eight separate QKV workers, 32 native paged-attention workers, sixteen SwiGLU workers, sixteen layer-normalization workers, two RoPE workers, two KV writers and two collective workers. Sixteen terminal head and eight terminal norm workers complete the 110-core layout. Attention and the subsequent MLP remain dependency ordered. Gate and up remain a packed projection. Tested GU16 layouts lost, but an independently scheduled gate group and up group was not separately qualified; that remains an open design option. No simultaneous-function claim is made merely from having distinct groups.

The native DRAM-sharded matmul already uses triple-buffered weights, two-block look-ahead and transaction IDs1–3; its input path is double buffered. The original resident projection reader used two slots and waited after each block, beginning phase reads after its activation join. Reusing the native reader strategy repairs that disadvantage. The additional phase-prefix reads start before the resident activation join and are separately measured. Neither buffering nor asynchronous reads are claimed as absent from the native baseline.

The selected changes are:

- Read weights through three aliased L1 buffer slots and explicit pipelined transaction IDs, with two blocks outstanding. A two-block prefix enters each projection's own buffers before its activation join; the ordinary reader skips those same reads. Initial reads and every refill occur inside every layer/token, including the first token. No cross-request weight residency or uncharged prefetch is assumed.
- Place the eight projection workers at native bank-near reader coordinates, preserve bank/column order when scattering outputs, and coalesce contiguous activation input reads. Dependent A/B/partial buffers alias the same storage; each phase restores its original logical ring capacity.
- Use wider projection subblocks and 16-row metadata with the original page stride and arithmetic order. Initialize constant and padding scratch once per resident invocation, retaining required clears for every new invocation and prefill transition.
- Use a specialized seven-column Blackhole LoFi MVMUL for packed gate/up. It reads the original BFP4 weights, keeps K-block 8 and all original BF16 L1 partial-rounding boundaries. Compact eight-row A storage feeds 16-row partial/output geometry. UNPACK and MATH `tensix_sync` calls are required: without them, first-subblock errors depended on scheduling.
- Transfer the first 1024-byte FP32 face of layer norm statistics, which contains all sixteen row statistics. Zero the other faces every invocation. The terminal norm path is unchanged.
- Choose terminal head/norm placement before relocating half of the attention workers. Attention still has the native 32 ranks, chunk 256 and reduction geometry. A controlled 16-attention-worker test with head placement held fixed saved about28 microseconds; letting its head move saved another102 microseconds. The general candidate captures most of that placement benefit while retaining the original attention arithmetic.
- Retain both global joins per layer, but use reset/sense counters and multicast release. Local RISC rendezvous, CB reset, fabric drain/close and packet-header lifetime remain explicit.

The selected command flags are:

```text
--projection-reader pipelined --projection-buffers 3 --wide-subblocks
--bounded-layer-barrier --alias-projection-cbs --projection-placement dram
--scratch-init-once all --early-weight-blocks 2 --multicast-layer-barrier
--coalesce-input --projection-tile-height 16 --norm-tile-height 16
--head-placement select --custom-gu --norm-stats-face
--attention-placement head_priority --require-exact
```

The bounded short-context variant replaces `--attention-placement head_priority` with `--attention-workers 16`. Its exact qualification covers the fixed 128-start, 32/256-output workloads only. It is not a general attention configuration: reduced pools failed long-context numerical checks. No automatic context dispatcher or serving integration was added.

## Correctness, lifetime and limits

All selected qualification runs compare every teacher logit and teacher step, all request-touched pages including padding in all 64 K/V tensors, greedy outputs and repeated generations exactly against the matched native reference. The fixed prompt, sampling settings (top_k 1, top_p 0, temperature 1, seed 42), precision policy and teacher stream are unchanged. Focused real-layer tests also exercise all allocated pages/unused sentinels, remapped captured page tables, positions around127/128 and255/256, inactive -1 warmups and replay. Four-distinct-layer tests use layers 0/10/20/31.

This is fixed-prompt, batch-one agreement, not broad model quality qualification. The separate BF16 Hugging Face diagnostic failed the provisional PCC 0.99 target for both native and the exact prototype: PCC 0.977533 at 128 and0.984128 at 2048 in the supplied evidence. The matched TT result does not convert that failure into a pass. Long-context attention reassociation experiments failed their predeclared stronger gates; no failed tolerance was loosened.

KV allocations and their address table must remain alive and stable while any resident trace references them. Page-table contents and positions remain device inputs. Outputs alias reusable scratch and must be consumed before reuse. Release traces before switching implementations. Prefill stays native, and the complete body borrows native workspaces to coexist with long-context prefill.

The selected barrier resets arrivals before publishing a toggling sense release, so the old layer-arrival wrap is not inherited. Other collective counters have an explicit lifetime budget: at most 2^26 phases, or1,048,576 complete resident invocations in the bounded mode. The legacy path is conservatively limited to2^25 phases / 524,288 complete invocations. Direct calls, inactive warmups, capture and generator trace replay reserve the budget. Raw trace callers must call `reserve_invocations`; at the limit, release traces and rebuild the body and global semaphores. Resetting only the host counter is unsafe. This is a software bound, not a million-token soak-test claim.

Separate full8192 worker-core Watcher passes the exact teacher/KV/greedy gate and logs checks on all four chips. ETH Watcher instrumentation is excluded because its kernel exceeds the available configuration space. Final selective profiles (128 main/O/GU/down,8192 main-v2) each cover three complete windows, all four chips and35 operations per chip/window, with zero unmatched selected markers. The8192 retry uses16,000-operation capacity; the failed4,000-operation capture is retained but excluded.

Matched same-core BRISC firmware intervals cover110 cores per chip/window. The median maximum model-program interval across three windows ranges7.101–7.113ms across the four chips at128, and8.180–8.190ms at8192. These instrumented model-only intervals exclude separate sampling and host overhead. Chip2 has an approximately22.25-second cross-core timestamp offset, so an all-chip firmware wall span is invalid. No overlapping worker durations are summed; even the approximately7ms LM-HEAD-MATH zone includes waiting for the layer loop and is not7ms of head compute.

## Storage, code and bandwidth model

Per chip, selected projection weight payload is 6,684,672 bytes QKV +4,456,448 O +16,515,072 gate/up +15,597,568 down =43,253,760 bytes/layer. Across32 layers plus142,606,336 head bytes, this is1,526,726,656 bytes/token, excluding norm, embedding, KV and metadata. Dividing by approximately7.5ms gives approximately204GB/s of useful model-weight payload per chip. This is a latency-derived payload rate, not a measured DRAM-bus bandwidth or controller utilization counter. Four-chip times are wall times and must not be divided by four.

Blackhole provides1.5MiB L1/core. One layer's43.3MB weights fit only in aggregate storage; a whole layer does not fit in the selected worker's buffers, and all model weights do not fit on chip. The selected design uses bounded tile-block staging. Full QKV/O staging on otherwise idle head workers fit but added roughly11MB/chip/layer of remote-L1 copies and lost latency. Read-ahead depth and placement were therefore measured rather than maximized.

The selected descriptor's largest static CB allocation is991,232 bytes/head core; shared projection cores use592,384 bytes. Loop state is4096 bytes/core. Pinned CB views overlap allocated tensors and must not be added again. Allocator snapshots after warmup are not peak execution memory measurements. One selected long-run snapshot reports L1 allocator195,072 bytes/bank allocated,1,249,664 free; static program CBs, firmware, code, configuration and stacks are additional. Full8192 prefill/capture/decode qualification establishes that the selected configuration fits that tested workload, not all combinations of experimental flags.

The selected8192 kernel-code audit covers 60 ELFs across four resident program IDs. Maximum text bytes by RISC are BRISC 12,000; NCRISC 7,784; TRISC0 14,788; TRISC1 13,652; TRISC2 11,180. These are allocated text sections, not complete per-core memory peaks. An eight-buffer QKV experiment exceeded L1 by24,448 bytes and was rejected after bounded recovery.

## Evidence that changed the design

The measurements support a selective shared pool, not a universal all-operation pool. Moving QKV onto the eight later projection workers reduced participating layer workers 86→78 but lost approximately120 microseconds in the controlled composition (8.6260 versus8.5062ms), with exact output. Extra scatter/scheduling costs are plausible explanations, not isolated attribution. Separate QKV is retained.

More prefetch was often worse. Norm-helper down staging lost over2ms; idle-head QKV/O staging produced10.589ms versus8.506ms. Three outstanding blocks with three slots produced8.2435ms versus approximately7.64ms, and four slots/three outstanding blocks7.7050ms. Each was charged its refill and communication costs. The selected own-buffer two-block prefix avoids the helper copy and has separately verified same-core completion before activation joins. In final128 captures, median prefetch duration/completion lead before the subsequent read phase is QKV5.92/9.07us, O4.40/60.30us, GU4.38/20.49us and down4.15/12.70us; each covers3,072 intervals (32 layers×8 readers×4 chips×3 windows). All leads are positive. This demonstrates early read completion before the activation-dependent continuation, not a DRAM-controller utilization measurement. Such intervals overlap across cores and are not additive wall-clock savings.

CustomGU16 component arrangements were159.01us with native bank rows and157.76us with split bank rows versus144.73us for GU8; all were exact. That rejects these tested layouts, not all possible16-worker designs. Custom O, zero-padded custom down, full DST, custom QKV, cached layer tables and broader head-placement searches were also implemented and measured; none improved the selected composition. Tiny apparent wins such as inline CB reset were not promoted without evidence beyond noise.

A single-global-join-per-layer experiment used alternating local semaphore banks and passed short full-model and four-layer Watcher tests. It hung at full context2048 after model readiness. Live triage across all four chips was saved before stopping only its child process group. Subsequent source inspection found a concrete address mismatch consistent with the captured attention waits: legacy `get_semaphore(id)` polls the new bank, while `Semaphore<>` sends through templated `get_semaphore<core_type>(id)`, bypassing the function-like macro. The short four-layer fixture uses positions 127/128/129 and does not exercise this reduction tree. No repaired version was hardware-qualified; other protocol races remain unassessed. A bounded reset plus health/connectivity/ring checks passed, and the otherwise identical placement candidate with both joins passed2048. `--single-layer-barrier` is rejected before hardware setup in the delivered source; its implementation and failing checkpoint remain available for investigation. This host-side guard was added after the measurement freeze and does not affect the measured selected/default configurations. The four-layer success did not justify trusting full 32-layer behavior.

Other failures and repairs are retained in WORK_LOG.md: malformed helper-publication alignment, profiler marker collisions/truncated captures, ETH Watcher code-size overflow, and a custom-down Watcher alias-boundary violation. The latter was fixed by splitting the DMA at the inactive alias boundary while retaining all bytes and Watcher checks; repaired custom down was exact but slower. Failed or incomplete captures do not support the latency claims.

## Reproduction and preservation

Source, toolchain and loaded runtime were matched: Clang20.1.8, CMake4.0.2, SFPI7.80.0[956], Python3.12.3, Torch2.11CPU and Transformers5.12.1; libtt_metal BuildID `467a3c083d79401b9b46f83db7a46430ca790654`. Both `tar` and `tt_pybinds` components were installed. Builds, virtual environments and kernel caches remained on local disk. Model weights remained in the specified model snapshot and were never committed.

On this worker, `source /home/moconnor/llama-minlat-114624/artifacts/run-env.sh` reconstructs the environment. Device commands require a newly verified exclusive allocation; job 114624 grants no access after 2026-09-23 17:12:44UTC. Adapt the checked runner's job/host/expiry to the new handoff before hardware work. Start with matching loaded/source BuildIDs, four-chip health/connectivity and ring validation. Do not simply reuse this job's runner after expiry.

The exact serial commands and ordering are in durable `artifacts/final-paired-plan.json`; selected flags in `final-candidate-flags.json`; every process has command/source/allocation/log records and `*-result/result.json` plus tensor evidence. Run the offline summary with:

```bash
python -m models.demos.llama31_8b_qb2.tests.summarize_experiment2 \
  --artifacts /data/moconnor/llama-minlat-114624/artifacts \
  --output /tmp/experiment2-benchmark-summary.json
```

Durable root is `/data/moconnor/llama-minlat-114624`. The verified incremental `checkpoint-current.bundle` requires the starting f776 commit; the original source/evidence mirror is read-only at `/data/moconnor/llama-megakernel-113796`. Compressed, integrity-tested profiler archives and SHA256 manifests preserve raw captures separately. No credentials, weights, build outputs or caches enter Git. No push, force push, merge, PR, firmware change, reboot or allocation extension/release was performed.

The measured source and final results are committed. The final documentation checkpoint and evidence-transfer verification are recorded in the durable PARENT_CHECKPOINT.md and artifacts/final-durable-verification.json after preservation. The next useful investigation is the residual terminal-head/attention placement interaction and critical-path activation delivery, with the original 32-worker numerical geometry held fixed. The single-barrier implementation must map both semaphore APIs consistently and pass long-context/replay qualification before any performance use.
