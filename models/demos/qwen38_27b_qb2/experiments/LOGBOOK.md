# Qwen3.8-27B Galaxy experiment logbook

This is the ongoing record of the single-Galaxy implementation and qualification
work. Times are UTC. Entries before this logbook was created are backfilled from
receipts and Git history; a publication time is not an experiment start time.
Missing execution times are explicitly left unknown. Keep failures and rejected
candidates. Append corrections instead of silently rewriting an old result.

The [evidence index](evidence-index.json) inventories published receipts with
SHA256, recorded status, timestamps where present, and JUnit outcomes. The
[change history](change-history.json) supplies exact revisions and authored and
committed times for the implementation history. These complement the narrative:
a green unit test, a clean experiment exit, and model qualification are different
claims. For example, the placement experiment completed while some candidates
failed its numerical gate.

## Objective and constraints

- Implement the updated single-Galaxy Decode Kernel Plan and pass its reference
  evaluations. Latest local plan extraction is
  `tt-blaze/local-notes/qwen38-27b-plan-updated-20261006.txt`; PDF SHA256 begins
  `19d89a`. The original artifact is
  <https://claude.ai/artifact/4gXpveXRCaEoaS9uCxa3zD>.
- Prioritize output throughput while conversations are at 32K, 128K, and 256K
  total context, with a 2,680 output tok/s/Galaxy target. Those lengths do not
  mean generating that many output tokens in each benchmark.
- Tune bandwidth before reducing precision. Current experiments preserve BFP4
  projection weights, BFP8 KV, BF16 activations, FP32 recurrent state, and the
  explicit accurate-attention policy. Runtime defaults are not auto-promoted.
- Measure one TP4 replica first, then physically execute eight replicas. An
  eightfold projection is never labeled measured Galaxy throughput.
- Reversible host-local disk/RAM changes only; preserve checkpoints, native
  installations, firmware and NFS. Hardware is serialized through
  `/tmp/tt-device.lock`; long jobs are persistent and have deadlines.

## Reproduction locations and pins

| Item | Value |
|---|---|
| Development branch | `anatarajan/qwen38-long-context-throughput-20261007` |
| Local worktree | `/private/tmp/tt-metal-qwen38-long-context` |
| Galaxy | `ttuser@10.228.203.98` |
| Host runtime | `/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006` |
| Host evidence and immutable source snapshots | `/home/ttuser/qwen38-artifacts-20261007` |
| Native Metal | `a08819ddbe23077f8037d3802303939064868ff6` |
| Weights | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Host-local checkpoint | `checkpoint-pinned-1d4bf0f2` under host evidence root |
| Measured worker grid | 12 x 10; do not assume 11 x 10 from other Blackhole systems |
| Serving plugin used in historical evaluation | `b7e4292e4193cba20abe9c7c68ce489201b2e36b` |

Source snapshots and launch JSON files pin the exact experiment, including
uncommitted patches. A source-lineage revision is not a claim that every file in
an isolated snapshot is a complete checkout of that revision. The per-file
hashes in each receipt remain authoritative. Stop only an owned unit to cancel
a run; retain its outputs. A quiet log or SSH timeout does not establish that a
job stopped: inspect its service and process before recovery.

## Timeline: bring-up and measurement fixes

| Time / evidence | Experiment or change | Outcome and resulting action |
|---|---|---|
| Oct 6; exact first-attempt time not retained here | Baseline full-model TP4 bring-up | Initial inherited 300-second pytest deadline expired while loading weights. Extended the bounded model test; not an inference failure. |
| Oct 6; [baseline-v2](../galaxy-evidence/baseline-v2) | All 64 layers, prefill, device sampling and traced decode | Repeatable 128-token output. Warm B1 approximately 38.89 tok/s/user, TPOT 25.72 ms, 63-token prompt. Fixed-length timing continued past EOS; this was not an EOS/quality test. |
| Oct 6-7; [replica failure](../galaxy-evidence/replicas-v1-failed.xml) | Eight-replica bring-up | Tokenizer returned `BatchEncoding`; JSON logging obscured the original type error. Normalize token IDs before device opens and guarantee cleanup. |
| Oct 6-7; [fabric attempts](../galaxy-evidence/fabric-v1-failed.txt) | Torus neighbor tests | First filter missed required `name.` prefix. Corrected filter ran all eight variants. Wrapper then exited 1 because its EXIT cleanup ended with a false conditional; fixed that separately without relabeling the historical service as successful. |
| Oct 7 03:08-05:39; [initial sweep](../galaxy-evidence/perf-sweep-tp4-v1) | TP4 input-length/concurrency sweep | Published measured input/output/TTFT curves. This predates accurate attention and the single-step GDN integration; comparisons must retain that distinction. |
| Oct 7; [eight-replicas-v2](../galaxy-evidence/eight-replicas-v2) | Compare isolated/concurrent replica timing | Tokens matched, but sequential host completion observations inflated fast replicas' times. Added independent completion observers; retained the failed measurement. |
| Oct 7 07:32; [eight-replicas-v3](../galaxy-evidence/eight-replicas-v3) | Corrected physical eight-replica G0 test | All 80 sequences matched; maximum median TPOT increase 0.00584%. Aggregate 288.164 tok/s at B1/short context. Does not qualify later policies or long-context scaling. |
| Oct 7; [layer-profile-v3](../galaxy-evidence/layer-profile-v3), [recovery](../galaxy-evidence/profile-recovery-v1) | Reduced real-layer P0 profiling and export recovery | Device cases ran; teardown ordering and profiler export failed in successive attempts. Generic Tracy import hit a 128-GiB cap on a 67.5-GB host CSV. Bounded CPU recovery found only three of six signposted windows and correctly rejected the capture. No complete stage attribution is qualified. |
| Oct 7; [storage recovery](../galaxy-evidence/storage-recovery-20261007) | Recover checkpoint/runtime storage after profiling pressure | Recovered and verified the pinned host-local checkpoint. Preserve hashes and recovery receipts; no original weights or NFS changes. |
| Oct 7 08:47:19-09:15:17; [deployment](../galaxy-evidence/galaxy-serving-v9/deployment.json) | Full 198-question GPQA-D through eight vLLM engines | **171/198 = 86.36%**, below the unchanged 89.2% gate (177 correct). Five truncated, all incorrect. 1,678.39 s, concurrency 128, 32K output budget. API checks passed. No causal attribution of the score gap to a kernel yet. |
| Oct 7 by 09:23:51; [HTTP sweep](../galaxy-evidence/galaxy-serving-v9/http-sweep/receipt.json) | Client throughput after GPQA | Nine cells completed, then `ReadError` at 8K/concurrency 128. Long-context cells were not run; deployment receipt remains failed. |

## Timeline: numerical diagnosis and GDN optimization

| Time / evidence | Experiment or change | Outcome and resulting action |
|---|---|---|
| Oct 7 06:41; [GDN v1](../galaxy-evidence/gdn-step-candidate-v1) | Standalone in-place FP32 recurrence | Correctness passed; missed P1 latency target. Kept candidate isolated. |
| Oct 7 by 08:05; [GDN v3](../galaxy-evidence/gdn-step-candidate-v3) | Split value columns 1/2/4 ways; overlap state writeback | B1 recurrence 66.67 -> 26.01 us with four partitions. Smaller/inconsistent large-batch gains. Accuracy, long-horizon and allocation rebinding passed; P1 targets still missed. |
| Oct 7 08:04; [attention tuning v4](../galaxy-evidence/attention-tuning-v4) | Native attention numerical baseline | 8K/B1 relative RMS 2.70035% exceeded the unchanged 2% gate. Did not tune speed around a failing baseline. |
| Oct 7 14:45-14:55; [precision v1](../galaxy-evidence/attention-precision-v1), [v2](../galaxy-evidence/attention-precision-v2) | HiFi4/FP32/accurate-exp comparison | v1 used an unsupported Python config alias. Corrected v2 ran all 12 comparisons; every mode still exceeded the error limit. Higher fidelity alone was insufficient. |
| Oct 7; [buffering results](../galaxy-evidence/kernel-diagnostics-v1/README.md) | One versus two GDN input work items buffered | All 30 variants and long-horizon/rebinding checks passed. Four-way split B8/B16/B32/B64 latency fell 25.4%/29.2%/31.0%/32.1%. Still above P1 target, before model preparation overhead. |
| Oct 7 15:10-15:24; [attention diagnosis](../galaxy-evidence/attention-gdn-integration-v1/README.md) | Chunk/core sweep, full-tile controls, accurate exp | All 48 approximate-math variants failed. Half-tile compute forces approximate exp despite the flag. Full tile plus accurate exp and chunk 256 passed the original numerical gates, including near 256K. No tolerance relaxation. |
| Oct 7 15:25-15:47; [model adapter](../galaxy-evidence/attention-gdn-integration-v1/README.md) | GDN boundary including preparation/layout, plus native control | Candidate passed every batch. Native control drifted beyond strict per-head 0.5% output RMS after 64 steps. Candidate improved large-batch boundary timing but regressed B1. Native-control failure alone does not explain GPQA. Attention wrapper also needed `tuple(Shape)` before slicing. |
| Oct 7 17:08-17:24; [real-weight layer attempts](../galaxy-evidence/gdn-model-integration-v1/README.md) | Integrate candidate with actual layer-0 operands | v1 failed B32 cancellation-sensitive heads. v2 isolated error to Q/K preparation, not recurrence on the supplied operands. Fused FP32 normalization fixed it; intermediate v3 used the wrong scalar-init API. v4 passed all B1/8/16/32 gates without changing precision. |
| Oct 7 17:40 and subsequent smoke; [kernel validation](../galaxy-evidence/gdn-model-integration-v1/gdn-kernel-validation-v1) | 4,096-step fused-normalization soak and 64-layer repeatability | Passed; final state/output relative RMS 3.11e-7/4.91e-7. Warm B1 34.97 tok/s, 69.66 ms TTFT. All P1 standalone latency targets still missed. This was not online-eval qualification. |

The real-layer v4 candidate reduced the measured GDN block latency by
29.1% at B16 and 40.6% at B32; B1/B8 regressed. Its state remains FP32. The
normalization/gating epilogue, projection, MLP, and full model each need their
own accounting; do not add overlapping speedups or substitute block latency
for model TPOT.

## Timeline: full-model capacity and bandwidth

| Time / evidence | Experiment or change | Outcome and resulting action |
|---|---|---|
| Oct 7 17:54-19:21; [native sweep v1](../galaxy-evidence/gdn-sweep-recovery-v1/baseline-v1.json) | Native accurate-attention full-model sweep | Twelve cells measured, then B32/32K prefill allocation failed. Each bank needed 160 MiB contiguous versus a 120-MiB largest block. |
| Oct 7 21:50-21:56; [fresh B32](../galaxy-evidence/gdn-sweep-recovery-v1/fresh-b32-v1.json) | Same B32/32K configuration in a fresh process | Also failed during prefill. Prior geometry was not necessary for this failure. Bound prefill chunk working memory instead of declaring a decode capacity limit. |
| Oct 7; [recovery attempts](../galaxy-evidence/gdn-sweep-recovery-v1/README.md) | Resume only verified cells after clean allocator OOM | Startup v2 lacked tokenizer baseline fixture; v3 omitted `pytest.ini` and collected outside source. Fixed snapshot dependencies and explicit root/config. No hardware result was credited to failed preparation. Only clean allocator failures permit restart; other failures stop. |
| Oct 7 23:18 native / Oct 8 00:59 candidate; [matched comparison](../galaxy-evidence/gdn-matched-comparison-v4) | Matched native vs single-step GDN, accurate attention, all 64 layers | Each completed 24 measured cells, one OOM and ten guards. At 32K/B16: **209.28 -> 259.22 output tok/s (+23.86%)**. At 128K/B8: 123.58 -> 123.90 (+0.26%); near256K/B4: 65.68 -> 64.74 (-1.44%). Sequential comparison; sub-percent differences are not established wins. |
| Oct 8 00:59-02:15; [capacity v1](../galaxy-evidence/long-context-capacity-v1/results/capacity/sweep.json) | Prefill budget 32,768 tokens, larger cache pool | B32/32K now passes: **250.02 output tok/s** native. Fresh-process B8/262016 passes: **89.98 output tok/s**, 11.25 tok/s/user. Changed prefill schedule is not reference-eval qualified. |
| Same capacity v1 | B16/128K after earlier request geometries | Cache allocation failed before prefill: 71,372,800 B needed per bank; 637,281,472 B free, but largest block only 46,448,640 B. This proves fragmentation at that point, not a cold-process capacity limit. Queue a fresh process per case. |
| Oct 8 02:16; [grid v1 failures](../galaxy-evidence/attention-grid-v2/README.md) | Placement followed by reader experiment | Placement asserted 11 x 10 before attention. Hardware reports 12 x 10. Reader then correctly rejected the failed dependency. Fixed runtime-grid handling; retained both failed attempts. |
| Oct 8 04:03-04:11; [placement v2](../galaxy-evidence/attention-placement-v2/placement.json.gz) | Native, row-major 64/80, FlashMLA-inspired 64, outer-column 80; baseline repeated | Thirty-six cases completed cleanly. Passing best/native speed ratios: 1.0456 at 32K/B16, 1.0447 at 128K/B8, 1.0061 near256K/B4, 1.0182 at 32K/B32. **All four alternative layouts failed the numerical gate at 128K/B16 and near256K/B8.** Native passed. No model promotion. |
| Oct 8 04:14-04:21; [reader v2](../galaxy-evidence/attention-reader-v2/queue.json) | Native, KV read barriers 4/8/16, native repeat; isolated kernel overlays/JIT caches | All numerical and clean-close checks passed. **Native was fastest in every geometry.** Larger thresholds slowed calls by up to about 23%. Baseline drift stayed below 0.02%. Reject these reader changes. |
| Oct 8 04:28:01 launch; [capacity pairs](../galaxy-evidence/capacity-pairs-launch-v1.json) | Fresh full-model process for each geometry and recurrence | Persistent `qwen38-capacity-pairs-v1-20261007.service`: native then single-step at 128K/B16, 32K/B32, and 262016/B8. Six runs, unchanged precision, 32K prefill budget. 254 CPU tests plus 40 subtests and historical one-cell comparison/render passed before launch. Running snapshot is not a pass receipt. |

### What the bandwidth numbers mean

Effective attention bandwidth is useful causal KV bytes / traced call time,
including output conversion when a candidate changes its layout. It is not
hardware DRAM-counter utilization. Against the assumed 512 GB/s/chip ceiling:

| Context / users per TP4 | Native GB/s | Best passing GB/s | Ceiling fraction, native -> candidate |
|---|---:|---:|---:|
| 32K / 16 | 349.0 | 365.0 | 68.2% -> 71.3% |
| 128K / 8 | 365.6 | 381.8 | 71.4% -> 74.6% |
| 262016 / 4 | about 377 | about 380 | about 74% |
| 32K / 32 | about 378 | 385.0 | about 74% -> 75.2% |

The 128K/B16 and 262016/B8 alternative-layout timings are retained but excluded
from improvements because they failed numerical acceptance. KV remains bank
interleaved; moving workers does not make reads bank-local. Full-model timing
includes weights, GDN state, compute, collectives and preparation; its traffic
roofline fraction is a separate metric.

## Reference research and simulator setup

| Recorded Oct 8 UTC | Finding and decision |
|---|---|
| [Native/reference kernel survey](REFERENCE-KERNELS.md) | Reviewed DRAM-sharded matmul reader placement, transaction-tagged prefetch, Flash-Decoding/FlashAttention and FLA GDN partitioning. Generic split-KV and double buffering already exist; avoid claiming them as new. |
| Thatch-cloud repository, `95589fb2ebdd7aa400e1daa89dcb3a8d457756bc` | Cloned read-only under `~/Documents/Tenstorrent.Blackhole-Qwen3.8-27B`. Their TP2 speculative verification differs from our TP4 single-token decode. No scripts executed or numerical paths imported. |
| Shared Q/K normalization | Their repeat saves about 1.2 ms in a roughly 68-ms verifier. Our candidate currently duplicates normalization 3 value heads x 4 partitions = 12 times per shared Q/K head. Next candidate should share our exact FP32 math and charge all preparation. **Not implemented/tested yet.** |
| Direct causal convolution windows | Their verifier saves about 3.5 ms but request timing is unstable. Our concatenation/layout preparation is a plausible removal target, not a proven gain. |
| Contiguous MLP weight blocks | Their pilot changes 48 x 576-B reads to one 27,648-B span; verifier 62.99 -> 60.27 ms. Adds about 3.01 GiB/card and setup latency; not directly appropriate for our constrained long-context KV pool. Audit existing DRAM-sharded reader transactions first. |
| Negative experiments and correction | Their additional bulk-read overlap regressed 0.60% on repeat; more MLP workers did not establish a request gain. Their claim that launch elimination cannot help was retracted because both arms ran the same image. Do not reuse that conclusion. |
| Attention KV sharing | Their multicast requires identical page-table rows among speculative entries. Independent users cannot share arbitrary KV this way; our TP4 single-KV-head geometry also excludes their query-head slicing optimization. |
| Official simulator clone | `https://github.com/tenstorrent/tt-sim.git` returned repository-not-found; resolved official repository to `https://github.com/tenstorrent/ttsim.git`. Cloned into `~/Documents/tt-sim` at `f6150d114139a0b5265b1497d996bc6139af2cad`. Read `AGENTS.md` and README. No simulator source modifications or experiments yet. |
| Local simulator compatibility | macOS host has Linux/aarch64 Docker available, about 8 GiB RAM. ttsim requires Linux runtime and a compatible Metal build for our kernels; cloning is not simulator qualification. Use for correctness/ownership experiments; keep bandwidth/performance claims tied to hardware. |

Reference report links and exact caveats are in
[Thatch review](THATCH-REVIEW.md). The simulator is an additional validation
tool, not a replacement for hardware, long-horizon state checks, or model evals.

## Remaining gates and next experiments

1. Finish the fresh-process capacity pairs; keep OOM, accuracy, and timing outcomes
   distinct. Then choose the useful batch/context operating points.
2. Implement shared FP32 Q/K preparation in an isolated candidate. Check real
   cancellation-sensitive heads, changed-input replay, immutable inputs,
   persistent addresses, 4,096-step state drift, and total adapter/model time.
3. Establish compact reader/compute/writer attribution and calibrated bandwidth
   baselines. P0 complete stage accounting and the P1 latency target remain open.
4. Investigate bank-aware state/KV layout and direct convolution windows. Do not
   lower precision or promote the failing long-context placement candidates.
5. Complete B64 projection support and required fusions, prefill targets, and
   physical full-Galaxy sweeps. Component B64 recurrence is not full-model B64.
6. Requalify the final policy through G0, serving/API and the required reference
   evaluations. Latest published full GPQA is below gate; no later passing
   result is claimed. Terminal/SWE small inherited subsets do not meet full
   evaluation scope. The whole objective remains incomplete.

## Maintenance convention

For each new attempt append: UTC start/end, hypothesis, exact source and launch
receipt, changed knobs, precision, geometry, expected gate, measured outcome,
failure/recovery, next action and publication revision. Store raw receipts under
a new directory, update the evidence index, and push working progress. Record
running jobs with their service name and observed PID; update with terminal
evidence later. Never infer success from queue order or a missing log line.
