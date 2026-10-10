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
- Oct 8 priority clarification: 128K and 256K carry substantially more weight
  than 32K. This is not a strict no-regression rule: a tens-of-percent 32K gain
  may justify a few-percent loss at 128K/256K. Explicitly flag the measured
  gains/losses at each length, confidence/repeatability, and capacity effects.
  Prioritize placement and bandwidth work; keep precision investigation separate.
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

### Oct 8: CPU simulator accuracy experiment and revised priorities

- **04:47:18-04:47:23 UTC:** first CPU-only Blackhole simulator smoke failed
  before matmul because pinned Metal exposes `WormholeComputeKernelConfig`, not
  `BlackholeComputeKernelConfig`. This is a Python compatibility failure, not a
  BFP4 kernel failure. Retained the failed source and log.
- **04:49:04-04:49:05 UTC:** corrected smoke completed with one virtual device,
  finite output and clean close. The separate simulator inspector RPC was
  disabled to avoid colliding with the physical experiment's inspector port.
  No simulator arithmetic/error checks, native kernels or hardware settings were
  changed. Official ttsim **v1.11.2** Blackhole binary was digest-pinned; the
  existing native Metal build remains `a08819ddbe23077f8037d3802303939064868ff6`.
- **04:55:01-04:55:23 UTC:** completed **60** real-weight submatrix matmuls:
  ten slices covering linear-attention QKV, full-attention Q, MLP down/gate and
  LM head, each at BFP4/BFP8/BF16 and LoFi/HiFi4. BF16 inputs are seeded synthetic
  stimuli, not captured model activations. Separate host references isolate
  weight quantization, output error due to quantization, and matmul execution
  error. All outputs finite, single virtual chip, clean close. No task-accuracy
  acceptance threshold was asserted and no model precision was changed.
- BFP4/LoFi median total output relative RMS is **11.793%**, versus **11.803%**
  for quantization alone and **0.439%** execution error against quantized
  operands. BFP4/HiFi4 execution error falls to **0.167%**, but total error stays
  **11.804%**. Quantization dominates this sampled probe; this is not an
  explanation of the historical GPQA deficit or an end-to-end qualification.
  Full six-mode results, sources, failure evidence and reproduction command are
  in [the simulator report](../galaxy-evidence/bfp4-simulator-v1/README.md).
- Resource isolation: persistent simulator units use one CPU/8-GiB limits,
  separate JIT cache and explicit simulator selection. They do not acquire
  `/tmp/tt-device.lock` or open physical Galaxy devices. The existing 128K/B16
  full-model capacity run completed its first repeat and remained active when
  these simulator receipts were collected.
- Shared-Q/K preparation now has an **unvalidated local prototype**. No
  compilation, numerical or performance result is claimed for it, and it is not
  enabled in the model. Following the user's clarification, long-context
  placement, bank traffic and reduction work take precedence over this candidate.
- Next placement diagnosis must distinguish changes in physical location from
  changes in work partition/reduction depth. The earlier 64-/80-core alternatives
  also reduced cores per user versus native at 128K/B16 and 256K/B8. Equal-count
  alternatives had identical error patterns; that is a lead, not proof of the
  numerical cause. Preserve the accurate-exp/FP32 policy and the original gate;
  establish matched-work comparisons before claiming a placement improvement.
- **Subsequent reference review:** cloned `tenstorrent/tt-lab` at
  `e6baf562a72491ed9d594cddbdf42e0925b1f79e`, read its instructions and inspected
  quantization, exact proxy/device comparison, teacher-forced CPU/proxy checks
  and logit diagnostics. No build/model/device commands run from that repo.
  [Review and applicability](TT-LAB-REVIEW.md): use separate arithmetic and
  quality references; its active GPT-OSS expert path is BFP8 and its exponent
  selection/rounding differ from standard Metal, so its BFP4 helpers are not
  drop-in Qwen correctness or qualification evidence.

### Oct 8: long-context placement follow-up and plan audit

- **05:10 UTC:** fresh native 128K/B16 full-model run completed with clean
  device close. Three measurements agree on output hashes and give **142.06
  output tok/s per TP4**, 8.879 tok/s/user, 112.63-ms TPOT. The prior mixed-
  geometry allocation failure was not a cold-process capacity limit. The
  single-step comparison began next and has not yet supplied a paired uplift.
- **05:16:12 UTC:** launched persistent
  `qwen38-attention-placement-long-v1-20261008.service`. New source snapshot,
  unchanged native runtime and BFP8 KV precision. **263 CPU tests + 40 subtests
  pass**; service PID 2559056 was observed live waiting for `/tmp/tt-device.lock`
  behind the full-model run. No active hardware job was interrupted.
- New experiment: 78 synthetic TP4 cases, prioritizing 256K/B8 and 128K/B16.
  Compare 80-/96-core row-major and outside-column placements at equal work
  counts, explicit full-grid sharded-output controls, and 256/512-token chunks.
  Retain the original numerical tolerances; record exact output hashes and
  bracketed native timing. No speedup is claimed while queued. Source hashes,
  launch command, CPU JUnit and native 128K receipt are preserved in
  [placement-long-launch-v1](../galaxy-evidence/placement-long-launch-v1).
- User explicitly authorized a parallel BFP4 KV experiment. A separate agent
  runs only the CPU simulator with isolated cache/resource caps; physical
  performance work stays on BFP8 KV. This is distinct from the already completed
  BFP4 **weight** probe. Long-context KV results and compatibility differences
  will be appended with terminal receipts.
- Audited the requested P1/P2/overlap milestones against code and receipts.
  [Plan status](PLAN-STATUS.md) records that recurrence correctness passes but
  full epilogue fusion and P1 latency do not; P2/B64/resident buckets, complete
  stage attribution, prefill MFU, mixed-step overlap and reference evals remain
  open. The 4.7-ms fixed overhead and 85% DRAM numbers remain assumptions.
- The same status report records current measured-TP4 projections and the
  explicit BFP8 traffic model. At 256K, KV-only full-scan bandwidth is already
  an optimistic ~1.8K output tok/s/Galaxy bound at assumed 512 GB/s/chip, before
  weights/state. The 2,680 target there needs lower traffic per accepted token;
  it cannot be credited solely to better placement.

## Bounded profiling and completed KV simulator screen, Oct 8 UTC

- **05:28:56:** the parallel CPU KV simulator run closed cleanly. All 24 cases
  passed the existing execution gate against quantized operands. Across the
  18 long-context cases, total attention-output RMS was 1.281-1.861% for K8/V8,
  15.647-16.641% for K4/V4 and 11.116-11.288% for K8/V4. These synthetic results
  keep BFP8 as the current policy; real activations, logits and model evals
  are still required for a BFP4 suitability conclusion.
- Preserved the first long-run unsupported-SFPLOADMACRO failure. The successful
  run used the native explicit-instruction compiler fallback with checks intact;
  six short controls were bit-identical, but production instruction parity at
  long context is not proven. Published all original receipts/source snapshots,
  a formatted probe and a verifier: 17 artifact hashes, six smoke hashes and
  all 24 numerical gate results verified. See
  [the simulator report](../galaxy-evidence/bfp4-kv-simulator-v1/README.md).
- **05:40:46:** the bounded layer-profile harness passed 276 CPU tests plus
  40 subtests in 3.24 seconds. It forbids prefill, uses actual layers 0/3 with
  seeded populated state/KV, and permits only two warm calls plus one marked
  decode. Restored inputs must produce identical finite logits on all calls.
  Captures are independent for 256K/B8, 128K/B16, 32K/B16, 8K/B32 and 8K/B1,
  each with native and single-step recurrence and unchanged BFP8 KV/FP32 state.
- **05:42:41:** launched persistent
  `qwen38-bounded-layer-profile-v1-20261008.service`, waiting for the placement
  diagnostic's authoritative terminal state and clean receipt before taking
  the shared device lock. CPU revalidation passed in 2.92 seconds. The unit
  has a 12-hour maximum, 64-GiB memory and eight-CPU limits; each capture checks
  a 1-GiB file/4-GiB total export budget and 16-GiB free-space floor. This
  avoids repeating the previous 67.5-GB export failure. No existing job was
  stopped, and installed runtime/checkpoints remain untouched.
- The collector requires passing JUnit, exactly three matching outputs, clean
  device close, both decoder-layer windows and complete four-rank timings.
  Missing reader/writer/compute intervals reject the report. These intervals
  include waits and overlap; do not sum them or call them active compute time.
  This is warm eager attribution on synthetic caches, not a traced full-model
  P0 pass. Launch and CPU evidence are in
  [bounded-profile-launch-v1](../galaxy-evidence/bounded-profile-launch-v1).
- **05:44:42:** authoritative systemd polling confirmed the capacity-pair,
  placement and bounded-profile controllers all live. Capacity held the
  hardware lock; the other controllers were waiting. No new performance win
  is credited to a queued or running diagnostic.

## Paired gain, partial-query screen and priority change, Oct 8 UTC

- **05:53:09:** the 128K/B16 single-step model closed cleanly. Three-repetition
  native/candidate throughput is 142.063/163.359 output tok/s per TP4: **14.99%**
  uplift. Source/prompt/precision match except recurrence; hashes repeat within
  each arm but differ between arms. Prefill remains ~4044 input tok/s. The 1307
  tok/s Galaxy figure is an 8x projection, not measured scaling or an eval pass.
- **06:02:** started isolated accurate-partial-query simulator v1 after 285 CPU
  tests and 40 subtests passed. It stopped after the first passing full-tile case:
  the evidence collector expected compute includes in `kernel_includes.hpp`, but
  this runtime puts them in generated TRISC wrappers. Clean virtual-device close;
  source/log/result retained. Corrected the checker to require unpack/math/pack
  wrappers and exact overlay paths, preserving thresholds and original attempts.
- **06:08:27:** simulator v2 launched persistently after the same CPU suite passed.
  It completed both 16-case arms, exit 0. All candidate cases pass and eight partial
  outputs are bit-identical to full-tile controls. Six native partial cases fail.
  Candidate scaling/exp honors valid faces; BFP8 precision is unchanged. This is
  simulator-only, with explicit-instruction compatibility fallback; no hardware
  speedup or model qualification yet.
- **06:09:53:** all 78 placement cases completed with clean device close. Best
  passing attention-call gains: 32K/B16 +2.52%, 32K/B32 +1.94%, 128K/B16 +3.57%,
  256K/B8 +3.60%; native remains best at 256K/B4. Failed numerical candidates
  excluded. Useful KV bytes/time reaches ~70-76% of assumed peak, not measured
  DRAM counters. Full-model attention promotion remains gated.
- User revised priorities: **32K ISL primary, 16K second, 128K/256K still active
  secondary optimization targets**. Report throughput/TSU/TTFT/memory/accuracy
  tradeoffs rather than applying one policy globally.
- **06:16:39:** replaced the older profile queue while it was still waiting on
  flock. Freeze and process inspection verified no device worker had started;
  the active 32K capacity run was untouched. New persistent v2 profile order is
  32K/B16/B32, 16K/B16/B32, 128K/B16, 256K/B8, both recurrence variants. CPU
  gate: 276 tests plus 40 subtests. Export/resource limits unchanged. Live
  `Linger=yes` and active controller PIDs confirmed disconnect persistence.
- Inspected Blaze reload documentation at cached revision `9415da978b3` and
  streaming kernels at `75ae38aabb0`. Its 471 GB/s / 92% reference measures
  expert streaming matmuls; applying it to reload is explicitly a model
  assumption. Bank-adjacent readers, contiguous bank-local shards, larger
  packets and oldest-transaction waits are relevant candidate mechanisms.
  Qwen KV bank-local/pipelined reading is not implemented or tested yet.
- Collected and verified 73 original artifact hashes, matched pair inputs/source,
  78 terminal placement cases, 16 simulator candidates and eight partial/full
  output hash matches. See [follow-up evidence](../galaxy-evidence/optimization-followup-v1).

## Hardware queue and profiling wrapper correction, Oct 8 UTC

- **06:30:43:** native 32K/B32 full model completed, three repeats and clean
  close: 250.13 output tok/s per TP4, 7.817 tok/s/user, 127.93-ms TPOT.
  Prefill is 5184.79 input tok/s, TTFT p50 202.50 seconds. The single-step
  pair is still running; do not claim its uplift before the paired receipt.
- **06:30:46:** the first v2 profile failed before device work. The controller
  selected the Blaze safe-test wrapper, which lacks `--profile-ops`; pytest
  rejected the argument. This was a harness failure, not a hardware stall.
- Fixed selection to a profiling-capable Metal wrapper frozen with the source
  and covered by source hashes. Added shell-syntax and Tracy-help preflight
  before waiting for the device. Preserved the original failed log/receipt.
- **06:35:55:** launched v3 profile, same 12 captures and resource/export caps,
  after 276 CPU tests plus 40 subtests passed. Preflight succeeded and the job
  was verified waiting on the shared lock. No profiling timing claim yet.
- **06:40:24:** launched persistent physical partial-query diagnostic after
  295 CPU tests plus 40 subtests passed. It rechecks completed simulator
  evidence, matches the candidate header hash and uses production instructions
  without the simulator fallback. Ten geometries give 30 full/partial/full
  component cases, prioritizing 32K then 16K, retaining 128K/256K tuning.
  Original numerical gates, replay determinism, four ranks, input hashes and
  <=3% timing drift are required for a speedup. Model defaults remain unchanged.
- **06:43:27:** authoritative systemd and loginctl collection confirmed all
  three controllers live and `Linger=yes`. The hardware diagnostic/profile
  wait behind the healthy full-model run. Three full-model variants remained
  at the count check, roughly 1.5-2 hours of model work plus diagnostic
  contention. Total 2-4 hours is provisional until a bounded profile succeeds;
  the 14/12/5-hour service deadlines are safety caps, not ETAs.
- Published source and all prior results at `fba818e6acc`; verified the remote
  branch. New launch/source/CPU evidence, retained profile failure and the
  completed native 32K/B32 receipt are in
  [hardware-followup-launch-v1](../galaxy-evidence/hardware-followup-launch-v1).

## Physical attention completion and recovered profile, Oct 8 UTC

- **06:51:12:** matched 32K/B32 single-step model completed three repeats and
  clean close: 350.417 output tok/s per TP4, 10.951 tok/s/user, 91.320-ms TPOT.
  Native was 250.133 output tok/s: **40.092% uplift**. Prefill remains about
  5185 input tok/s. Eight-replica projection is 2803.34 output tok/s, not a
  physical Galaxy result or model-eval pass. The user clarified that the goal
  is maximum performance; 2680 is a checkpoint, not an optimization ceiling.
- **06:53:** v3 profile hardware test/export succeeded; its collector failed
  on missing TRISC times for data-movement-only ops. CSV source/hash lists and
  all three compute binary sizes prove these ops contain no compute kernel.
  Corrected accounting distinguishes proven non-applicability from missing
  measurements. Unknown/contradictory metadata still fails. Retained the failed
  receipt and log, and reanalyzed a copy without changing original evidence.
- **06:58:11:** physical partial-query test closed devices cleanly. All 30
  cases and ten bracketed comparisons passed; all candidate outputs are
  bit-identical to controls, with production instructions. Call throughput
  gains: 32K/B8/B16/B32 +4.28/+2.16/+0.85%; 16K +7.03/+3.48/+1.38%.
  128K/B16 and 256K/B4 regress 0.33/1.18%; retain context/batch policy choice.
  Full-model attention promotion and additive gains with placement are untested.
- **07:25:25:** frozen v4 profiler launched after 292 tests plus 40 subtests
  passed. Its source includes only the collector fix, preserving the current
  GDN kernel. A local pytest attempt lacked the dependency; remote validation
  passed. Recovered native 32K/B16 capture has all four ranks and 161 device-op
  rows per rank (96 data-movement-only). The corrected 12-capture queue uses
  the shared lock, 64-GiB/12-hour limits and unchanged export caps.
- Calculated current memory-traffic ceilings: 32K/B32 8654 output tok/s/Galaxy
  vs 2803 projection (32.4%); 128K/B16 2841 vs 1307 (46.0%). These use assumed
  peak bandwidth and omit extra traffic/compute/collectives. They are not
  measured DRAM utilization. Raw files and 21 verified original hashes are in
  [profile-recovery-and-throughput-v1](../galaxy-evidence/profile-recovery-and-throughput-v1).
- Remaining performance priorities: reduce GDN arithmetic/normalization and
  layout overhead, tune bank-local KV traffic, qualify B64 to amortize weights,
  and measure physical eight-replica scaling. No precision reduction promoted.

## Shared FP32 Q/K experiment queued, Oct 8 UTC

- **07:44:45:** v4's fresh native 32K/B16 profile completed and passed the
  corrected collector on all four ranks. The queue advanced to single-step.
  The last full-model capacity arm, 256K/B8 single-step, owns the device next;
  its weights finished loading at **07:49:53**. Native 256K/B8 completed at
  89.983 output tok/s per TP4 and 3687.09 prefill input tok/s, three repeats.
- Implemented an isolated shared-Q/K candidate: normalize each shared head
  once in FP32, keep persistent compact scratch and remove repeated Q/K
  expansion from the adapter. Recurrence math/state precision stay FP32.
  Extra preparation launch/traffic can erase benefits at small batch; no
  default change or measured gain is claimed before hardware results.
- **07:45:40:** `qwen38-gdn-shared-qk-v1-20261008.service` launched after
  **314 CPU tests plus 40 subtests** passed. Frozen source:
  `/home/ttuser/qwen38-artifacts-20261007/gdn-shared-qk-source-v1`.
  Five batches (32/16/8/64/1), fused/shared/fused adapter controls and 4096
  changing-input updates require original accuracy gates, bit-identical
  output/state, stable addresses and two alternating scratch allocations.
  Times include preparation and layouts. The unit is bounded to 32 GiB/5 h.
- **07:53:04:** `qwen38-gdn-shared-qk-layer-v1-20261008.service` launched
  persistently after the same CPU suite passed. It waits for the first job's
  authoritative success and matching kernel hashes. A qualified >=2% adapter
  gain at B16/B32 triggers six real-weight layer-0 comparisons, including
  convolution, gates, output normalization and projection. Otherwise it records
  the result and skips further hardware work. B16/B32 controls must retain
  FP32-reference accuracy and identical projected outputs. Unit: 48 GiB/6 h.
- **07:59:21:** all four controller PIDs were verified active. One of twelve
  profile captures is complete. Hardware work serializes through
  `/tmp/tt-device.lock`; the conditional layer job waits without opening a
  device. These are disconnect-persistent jobs, not reboot-resuming services.
- Source formatting/static hooks passed. Native compilation, shared numerical
  results and performance remain pending. Keep this runtime prototype out of
  the qualified source until those receipts are available; host snapshots and
  launch JSON pin all experimental files for reproduction.

## Remaining gates and next experiments

### Oct 8: speculative-decoding scope requested alongside bandwidth work

- Confirmed the installed checkpoint contains all 15 MTP tensors (424.7M
  parameters), using only its index and safetensors headers. No weight download,
  device opening, runtime change or new speculative hardware job.
- Refreshed Metal PR #55548: merged Oct 2; its merge is already an ancestor of
  the pinned native build. Reusable MTP, rejection sampling and multi-position
  SDPA exist, but the model loop is batch-one. Our custom model does not invoke
  them. The upstream recurrent factory also limits B*12 heads to 120 cores.
- Found the key high-concurrency blocker: existing fast projections accept 32
  token rows. K=3 needs four verify positions per user, so retaining B32 needs
  128-row verification; a naive 32-row port schedules only eight users/pass.
  This changes the priority from simply loading MTP to preserving batch width.
- Computed explicit memory/acceptance sensitivities. At B32/32K, four FP32
  snapshot planes plus MTP KV/weights add roughly 5.32 GiB/chip before workspace.
  Replaying selected recurrence from the original state is a lower-memory,
  extra-compute alternative. Neither implementation is qualified.
- A +33-69% throughput example assumes K=3, 70% conditional acceptance, the
  same user concurrency and a 1.5-1.9-step complete cycle. It is not a measured
  speedup or a forecast for a naive port. Lower acceptance or fewer active
  users can regress throughput. See [scope and gates](SPECULATIVE-DECODE-SCOPE.md)
  and its JSON receipt for sources, assumptions and arithmetic.
- User clarified the promotion rule: no speculative default in the main demo
  if total throughput decreases. Require matched offered concurrency and
  committed output tokens/s including every draft/verify/commit/sampling cost;
  a latency win at lower user count is insufficient. Keep ordinary fallback.
- **08:14 UTC:** all four existing systemd controller PIDs remained active.
  The 256K/B8 single-step model completed its first measured repetition;
  the other hardware jobs were still waiting. Bank-local read experiment
  remains under design and is not queued. No reboot or running-source edits.

1. Finish the fresh-process capacity pairs; keep OOM, accuracy, and timing outcomes
   distinct. Then choose the useful batch/context operating points.
2. Prioritize 32K, then 16K placement and bank-aware KV/state traffic; continue
   active 128K/256K tuning. Use compact
   reader/compute/writer attribution and calibrated bandwidth baselines. Diagnose
   the failing placement candidates' reduction geometry before promotion. Report
   32K/128K/256K tradeoffs together; do not automatically reject a worthwhile
   32K gain for a small, explicitly quantified long-context regression.
3. Validate shared FP32 Q/K preparation as a secondary isolated candidate. Check real
   cancellation-sensitive heads, changed-input replay, immutable inputs,
   persistent addresses, 4,096-step state drift, and total adapter/model time.
4. P0 complete stage accounting and the P1 latency target remain open. Investigate
   direct convolution windows after the priority bandwidth work. Do not lower
   precision or promote failing numerical candidates.
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

## Oct 8, 08:30-09:15 UTC: shared Q/K, physical bandwidth and full-model follow-up

Latest priority remains 32K ISL first, 16K second; 128K/256K remain active
secondary targets. No speculative default if aggregate committed throughput
regresses at matched offered concurrency.

- **08:30:22:** capacity pairs completed. Near256K/B8 native 89.983 versus
  single-step 89.831 output tok/s/TP4 (-0.169%); all repeats clean.
- **08:32:07:** shared-Q/K adapter passed all batches and 4,096 changing-input
  steps with bit-identical state/output. Adapter gains B32/B16 30.7%/26.2%;
  B1 regressed 2.8%, so retain the existing B1 path.
- **08:33:44:** real-weight GDN block passed. B32 1092.41 -> 973.30 us
  (1.122x); B16 730.64 -> 667.13 us (1.095x), projected output bit-identical.
- **08:42:50:** all 12 bounded profiles completed, including optimized paths.
  Recomputed every derived table from compressed raw CSVs locally. P0 full-model
  critical-path/roofline calibration still open.
- **08:44:36-08:45:34:** frozen bandwidth v2 launched and completed. CPU suite
  341 tests +40 subtests; hardware 42.57 s. Eleven variants and 36 timed cases,
  with full-byte tail/rebinding checks. Bank-adjacent bulk reads reached
  499.5-508.2 GB/s/chip; row bulk 340-343, tile-at-a-time about 190. This is raw
  read calibration, not an attention/model improvement. Four tags/eight-page
  packets nearly match the maximum; more buffering is not warranted yet.
- **Failed preflight:** v1 had ten fixture-call failures (missing expected error
  regex); device work never launched. Corrected tests and preserved failure
  receipt. Verified the user's interrupted staging attempt had not launched
  anything before retrying. Corrected an initial nonexistent copy-source path.
- **09:09:38:** shared-Q/K full-model comparison launched persistently after
  350 CPU tests +40 subtests. New policy preallocates scratch before traces,
  preserves fused B1/B2/B4, and leaves defaults unchanged. Control/candidate
  at 32K/B32, 16K/B32, 128K/B16, 262016/B8; 3 measured full-prefill repeats.
  Require identical output tokens, model sources, precision and concurrency.
  Unit `qwen38-gdn-shared-qk-model-v1-20261008.service`, PID 2860184 observed
  active after launch; 20-hour bound/128-GiB cap, host-local only, global lock.
- **Next:** measure bulk reader plus compute-worker delivery/backpressure;
  quantify its benefit against actual attention, whose bandwidth already exceeds
  the probe's simple interleaved control. Await shared-Q/K model results, then
  requalify useful policies through evals and physical Galaxy scaling.

Raw receipts, pins, failed preflight, compressed per-op profiles and reproduction
notes: [shared-qk-and-bandwidth-v1](../galaxy-evidence/shared-qk-and-bandwidth-v1/README.md).

## Oct 8, 13:10-15:53 UTC: shared-Q/K model pass and eight-replica qualification

- **13:10:41:** all four full-model pairs completed. Shared/control generated
  tokens match for every repeat. 32K/B32 350.416 -> 373.695 output tok/s/TP4
  (+6.64%); 16K/B32 402.164 -> 432.929 (+7.65%); 128K/B16 163.357 ->
  168.739 (+3.29%); near256K/B8 89.832 -> 91.303 (+1.64%). Prefill unchanged.
  Local recomputation verified raw repeats, precision/source parity and receipts.
  Eightfold 32K projection 2989.6 tok/s is still not physical Galaxy evidence.
- **15:46:56:** launched persistent eight-replica G0 on the passing immutable
  source, short-prompt B1, 3% concurrent/isolated TPOT gate. PID 3289234 observed
  live. B1 uses fused normalization fallback; do not claim B32 scaling from G0.
- **15:52:31:** queued full GPQA after successful G0/source/precision/JUnit checks,
  PID 3295587 observed live. All 198 questions, concurrency 128, unchanged
  32K output budget and T=1/top_p=.95/top_k=20/seed42. Gate remains .892.
  Added isolated serving-source selection and exit-after-eval mode to release
  hardware for subsequent experiments; no default model or native-install edits.
  Prefill budget is explicit 32768. CPU 350 tests +40 subtests; shell syntax
  and pre-commit passed. The evaluation has not run yet; no score claimed.
- Refreshed official model card: GPQA-D 89.2%, recommended thinking sampling
  matches. Complete official GPQA harness/budget is unspecified in that card.
- Host observation: 472 GiB available RAM, 56 GiB free on artifact disk before
  launch. Each service capped at 256 GiB; no NFS/checkpoint/firmware changes.
- User asked what remains: recorded prioritized bank-local delivery, GDN/layout
  fusion, B32 serving/B64 projection work, matmul/CCL tuning, prefill and overlap
  in [optimization backlog](OPTIMIZATION-BACKLOG.md). BFP4 and speculative
  defaults remain unqualified. Cross-core delivery prototype remains next work;
  no claim that the isolated read improvement has reached attention.

[Full-model results and launch receipts](../galaxy-evidence/shared-qk-full-model-v1/README.md).

## Oct 8, 16:03-16:30 UTC: remote-delivery prototype and G0 pass

- Revalidated existing jobs as live before proceeding; no restart or hardware
  interruption. Previous goal turn made progress by publishing the model gains.
- Implemented bank-local reader-to-consumer delivery using separate cores,
  cumulative ready/consumed credits, payload write completion before ready,
  and consumer completion before slot reuse. Local reader and remote receive
  storage each have bounded rings. Precision/model defaults remain unchanged.
- Hardware test covers three placements, depths 1/2/4, packets 8/15 pages;
  six variants, 24 timing cases including before/after raw-read controls.
  Correctness checks include two independent allocations, short tails, slow
  consumers, three trace replays and full-byte equality on all four chips.
  Large timings check packet markers; they do not imply full-byte validation
  at the largest volume or attention/production-page-table correctness.
- **16:19:51:** launched `qwen38-dram-delivery-v1-20261008.service`, PID 3360483
  observed live, waiting behind the GPQA unit. CPU 355 tests +40 subtests
  passed; installed semaphore descriptor construction passed without opening
  hardware. Twelve-hour service cap includes up to ten hours of dependency
  waiting, then a one-hour hardware/lock limit; 64-GiB/8-CPU bounds. No hardware
  performance or compilation pass for the new kernels has been claimed.
- **16:29:48:** G0 terminated cleanly after a passing JUnit test (42m45s).
  All eight physical TP4 groups pass output and concurrent timing checks.
  Worst concurrent/isolated TPOT ratio 1.000226 (+0.0226%). Scope remains B1
  short prompts; this does not measure B32 shared-Q/K Galaxy throughput.
- GPQA advanced to `starting` with exact-source qualification accepted.
  Delivery remains queued after it. All snapshots are on the allocated host's
  local artifact disk; no native install, checkpoint, firmware or NFS writes.
- Local pre-commit and source-hash comparison passed. SSH stdin collection and
  staging initially hit the local sandbox network restriction; retried via the
  required approval mechanism, with no duplicate jobs created.

[Delivery launch](../galaxy-evidence/dram-delivery-launch-v1/README.md),
[G0 pass](../galaxy-evidence/shared-qk-g0-pass-v1/README.md).

## Oct 8 evening / Oct 9 recovery: full qualification and release gates

- Full optimized GPQA at 32K finished 17:29:40 UTC: 142/198 (71.72%), 42
  truncations; below unchanged 89.2% gate. API checks passed. Compare historical
  native 171/198 with caution: multiple model changes separate those sources.
- 17:34: PATH repaired in isolated delivery/profile controllers; existing
  /usr/local/bin/tt-smi was missing from transient-unit PATH. No install needed.
- 17:36: delivery hardware pass. Read-only 499-508 GB/s/chip becomes 245-249
  including remote delivery/receiver work. Correctness success is not a model
  speedup. Extended placement/packet/ring/backpressure sweep prepared.
- 17:45: full 64-layer 32K/B32 test passed output repeatability, but profiler
  dropped markers and export hit the 1-GiB file cap. Full P0 remains incomplete;
  remaining profiling cases did not run. Preserve the failed capture. Fixed the
  helper's exit/signal race so it does not mask the primary capture error.
- SSH temporarily timed out to both hosts; no restart was inferred from that.
  Access recovered after the user requested retry. Existing jobs stayed owned
  by their persistent services. All new staging occurred after live validation.
- Tau3 setup uses pinned upstream source/lock and removable host-local venv.
  Root files were initially missing due to non-cone sparse checkout; repaired.
  Audio support is imported by the text runner; PortAudio/JACK/ALSA packages
  were extracted locally, not installed in the OS. Missing libasound2 was fixed.
- 18:05:34: overnight v1 launched. CPU 375 +40 subtests passed. Full GPQA at
  64K ran 18:17:36-19:15:12: **163/198 (82.32%)**, 15 truncations, 57m35s.
  Raw responses stay private on host for diagnosis; only hashes/scores published.
- Tau3 v1 failed all twelve processes before inference: sparse checkout lacked
  global user-simulation guidelines. This is no valid model score, not 0%
  demonstrated task accuracy. Restored data and added actual metadata preflight
  plus a null-score regression test for zero-call setup failures.
- Five HTTP cells completed before ReadError at 16K/128 clients. Native control
  and extended delivery were not run. The supervisor stopped its owned workers;
  server logs show subsequent SIGTERM, not a demonstrated prior device crash.
  A fresh-connection policy is a diagnostic change, not a proven root-cause fix.
- 03:28:38 Oct 9: resumed remaining control stages in a new immutable snapshot,
  qwen38-overnight-v2-20261009.service. CPU **376 +40 subtests** and actual Tau3
  metadata preflight passed; PID 917206 was verified live and loading native
  model layers after successful reset. G0 -> full 64K GPQA -> corrected Tau3 ->
  matching HTTP sweep -> 54 delivery variants are persistent and bounded.
  No already completed candidate score is relabeled or replaced.
- Release remains unqualified: GPQA below 177/198, no valid agentic score,
  incomplete serving sweep, and no pinned release image/tested Qwen Helm chart.
  Further kernel fusions and profiler repair are not in the unattended queue.

[Current queue and methodology](../galaxy-evidence/qualification-overnight-v2/README.md),
[delivery results](../galaxy-evidence/delivery-pass-v2/README.md),
[failed full profile](../galaxy-evidence/full-trace-attempt-v2/README.md).

## Oct 9, from 03:42 UTC: saved-response diagnosis and release preparation

- Revalidated PID 917206 as active before inspecting saved outputs. The native
  control continued loading; no live source, request budget or policy was edited.
- Matched all 198 private response hashes and token counts to the published
  scored receipts. All fifteen truncations hit exactly 65,536 output tokens,
  with 163-542 prompt tokens and no final answer. These are output-limit events,
  not exhaustion of 262,144 context. They remain incorrect in 163/198 (82.32%).
- Natural-stop subset is 163/183, not an alternate qualification score. The
  overall result would need fourteen more correct answers to meet 177/198.
- Exact duplicate 20-word tail sequences are below 0.246% for each truncated
  response. Private inspection of five tails shows continuing calculations and
  reconsideration; no claim they would finish correctly. A separate wrong
  natural-stop output is highly repetitive and has no final-answer content.
- Added a text-free response audit and six passing CPU regression tests. Local
  convenience environments lacked pytest (one older temporary env was gone),
  so tests ran in the existing remote task environment with isolated source and
  no device access. Raw GPQA examples remain private on the host.
- Installed HF Qwen3.5 reference and custom GDN both add 1e-6 to the squared
  norm before rsqrt; that suspected formula mismatch was not found.
- Created `/private/tmp/tt-inference-server-qwen38-release` from refreshed main
  `27699d0ac` on `anatarajan/qwen38-galaxy-release-20261009`. Main already has a
  P300X2 Qwen profile, but it uses a four-chip ring. It does not package the
  tested eight-process TP4 Linear Galaxy launch. Added digest rendering and an
  optional requirement that both the inference image and BusyBox-compatible
  init image are pinned. Six chart-render tests pass; the initial test helper
  computed its root one level too high, fixed before the successful run.
  The image build and Galaxy profile remain packaging work. No cluster changes.

[Audit evidence and reproduction](../galaxy-evidence/gpqa-response-audit-v1/README.md).

## Oct 9, 04:12-04:34 UTC: native G0 pass and image preparation

- Native recurrence G0 completed 04:12:55 UTC, one hardware test passing after
  43m18s including eight sequential model loads and warmup. All replica tokens
  matched. Concurrent/isolated TPOT ratios stayed within 0.0064% of one; the
  gate remains 3%. This short-prompt B1/replica test is not a long-context score.
- The same persistent evaluation controller started the standard eight-worker
  vLLM launch and full 64K-output GPQA around 04:22 UTC. At 04:29:52 it had
  completed 92/198, 87 correct and no length cutoffs. This is only a running
  snapshot; short/easier answers can finish first. No accuracy promotion.
- Prepared TTIS ModelSpec and Helm overlay with exact G0 physical chip groups,
  16 slots per replica, BFP8 KV, FP32 recurrence, 256K context, pinned checkpoint,
  and standard tool/reasoning parsers. Preserve the host launch's absent 5s
  Metal watchdog. Ensure EXTRA_MODELS_DIR exists before plugin discovery.
  Fixed the initial ModelSpec constructor to pass the engine string expected
  by its validator. The image import check uses the actual Qwen38ForCausalLM
  class; the plugin adds the TT architecture-name prefix.
- Strict source verification rejected three new standalone diagnostic files
  present at current HEAD but absent from the frozen qualification source.
  A new sparse, detached worktree at 0abdc3403f039c46becef335ad02db99237593f8
  matches every qualified model source hash. Kept verification strict. Initial
  --no-checkout worktree required index population; the sandboxed index write
  was retried through the approved escalation path. No prior worktree changed.
- TTIS preparation/Helm checks: 21 passed, Ruff passed, diff check passed.
  Published e0e05bad5361d7c170068b3ad7b4df27de192250 on the separate
  anatarajan/qwen38-galaxy-release-20261009 branch. Includes launch instructions,
  exact native/model/plugin/checkpoint pins and image verification. No claim
  that the image or final SJC3 deployment is qualified.
- Build host .34 has 33GiB root-disk free and more than 500GiB RAM available.
  Preserve existing images. New BuildKit cache uses a bounded, container-local
  executable tmpfs, avoiding root-disk exhaustion and /dev/shm's noexec flag.
  Outputs alone go to a new /dev/shm directory and need durable export.
- Build attempt v1 failed before compilation: unprivileged user namespaces
  disabled by host policy. No sysctl changed. v2 used a container-local mount
  capability without accelerator access and started BuildKit successfully, but
  the client inherited the rootless socket. After inspecting the healthy
  daemon and correct socket, explicitly stopped only that owned build service.
  v3 corrects BUILDKIT_HOST and starts actual image extraction/build. Its user
  service has a four-hour bound; container limits are 192GiB and 24 CPUs.
  Client disconnect does not cancel it. All failed-attempt logs remain intact.

[Native control and packaging receipts](../galaxy-evidence/qualification-overnight-v2/README.md).

## Oct 9, 04:35-04:55 UTC: image executor repair and next accuracy control

- Image build v3 reached Dockerfile execution, then failed on read-only nested
  cgroup creation. v4 proved a private-container remount left the 128-MiB test
  limit unchanged, but runc then failed its nested BPF device-filter query.
  Failed services and containers are terminal; logs are preserved. No host
  cgroup remount, user-namespace sysctl, firmware or accelerator operation.
- v5 uses BuildKit's rootless spec conversion plus explicit runc rootless
  cgroups inside the bounded container, keeping its process sandbox. A tiny
  pinned-BusyBox build passed root writes, UID-1000 chown and non-root access
  before the real image build. Source fetch succeeded and native CMake is
  configuring. qwen38-release-build-v5-20261009.service owns the persistent
  build; all earlier service names are historical. The final image still needs
  source/import checks, hardware qualification and durable registry export.
- Native GPQA reached 170/195 at 04:53:10 UTC with no cutoffs. At most 173/198
  is now possible, so this run cannot meet 177/198. Leave it running and retain
  the complete score. The early higher partial rate did not justify promotion.
- Prepared an unrun BFP8/HiFi2 LM-head control; existing selected weights use
  BFP4/LoFi at the head. Policy validation and comparison confirm that all 64
  decoder policies and all other numeric settings stay identical. This tests
  the combined head restoration, without claiming a root cause or gain. It
  needs its own G0 and full GPQA. It is not yet queued, and neither running
  evaluation nor image source was changed. Throughput/storage tradeoff must be
  measured before release selection.

## Oct 9, 05:09-05:30 UTC: complete control score, built artifact, queued head test

- Image v5 completed in 15m50s with source/import checks passing in both image
  stages. No hardware was exposed. The OCI archive is 6,010,501,632 bytes and
  has manifest digest `sha256:0b11f045bf089088a62b6e3c1aeb9b64cc72b74a632935f25203609b1e023579`.
  A separate bounded persistent job copied it from RAM to host disk, fsynced it
  and verified matching full-file checksums. It is unqualified and not in a
  registry. The original head policy remains fixed in the built image.
- Matched native control completed all 198 GPQA questions: 170 correct, one
  incorrect 65,536-token output cutoff, 27 naturally completed wrong answers.
  Measurement lasted 48m39s. All raw hashes/usage/finish reasons were audited;
  none exhausted the 262,144-token context. Full score 85.86% remains below
  177/198. Tau3 then started. No answer, threshold or denominator changed.
- Added an accuracy-only controller mode and a persistent follower. The latter
  waits on the exact service invocation and completed hardware/cleanup receipts;
  stale files and observation timeouts never establish that the device is free.
  Six tests exercise these boundaries and propagation of head precision through
  a fresh G0 plus full GPQA. It keeps the original full experiment queue intact.
- Head-control staging initially failed before CPU tests because native shared
  libraries were missing from LD_LIBRARY_PATH. The failure is preserved; using
  the existing task library directories fixed the preflight with no installation
  changes. 381 tests and 40 subtests passed; the one optional tokenizer skip was
  rerun separately against the pinned local tokenizer.
- `qwen38-accuracy-head-v1-20261009.service` is live and waiting after the current
  native/Tau3/HTTP/delivery queue. Source hashes, 18h total timeout, 12h wait,
  256-GiB memory limit and owned-process cleanup are recorded. It survives SSH
  disconnect, not host reboot. BFP8/HiFi2 head accuracy and cost remain unmeasured.

[Native results](../galaxy-evidence/qualification-overnight-v2/README.md),
[head queue](../galaxy-evidence/accuracy-head-v1/README.md),
[image artifact](../galaxy-evidence/image-build-v5/README.md).

## Oct 9, 05:39-06:00 UTC: bounded Tau3 result and queued CPU/image checks

- Corrected Tau3 completed 12/12 attempts in 32m36s, with 3 successes, 277
  validly parsed tool calls, zero malformed arguments and zero output-budget
  cutoffs. Four tasks hit the 20-minute task cap; task_026 hit its 300-second
  model-request timeout (raw row 10, 300.119971 seconds, 8K output budget).
  Three completed user-stop trials and one 60-step-limit trial scored zero.
  This self-simulated/graded pilot is not a matched published reference score;
  incomplete trials and manual conversation review remain explicit limitations.
- Native physical HTTP cells completed at 32K/C32 and 32K/C64. The 128-user
  warmup had an extended interval without visible tokens. Workers remained
  live, advanced from nine to sixteen active requests, then generated tokens
  across all eight engines at 05:56 UTC. No reset, restart or debugger attach
  was performed. Cold setup is a lead, not a proven cause; retain warmup timing.
- OCI inspection v1 assumed gzip and failed on zstd base layers. v2 dispatches
  by declared media type; all 46 layer hashes/sizes and manifest/config digests
  passed. Layer tar bytes total 18.91 GB uncompressed and 6.01 GB compressed.
  The build host lacks the conservative import space plus an 8-GiB reserve,
  so no existing image was removed and no import was attempted there.
- A bounded TLS source serves only the immutable image archive to allocated
  host .98. The certificate was copied through authenticated SSH; no private
  key or registry credential was transferred. The destination copy/import,
  no-device container checks and new CPU HF-head reference are queued behind
  both hardware queues in qwen38-post-head-cpu-v1-20261009.service. Their host
  load is not allowed to overlap the measurements. The image/head policy in
  all existing jobs remains unchanged. No result is claimed for the CPU probe.
- The CPU probe uses all original BF16 HF decoder weights, eight fixed public
  reference steps, and identical hidden states for head-only BFP4/BFP8 weight
  round trips. It rejects incomplete checkpoint loading and reports numerical
  sensitivity without including device fidelity or pretending to predict GPQA.

[Tau3](../galaxy-evidence/tau-pilot-v2/README.md),
[OCI inspection](../galaxy-evidence/image-build-v5/inspection/README.md),
[follow-up queue](../galaxy-evidence/post-head-cpu-v1/README.md).

### 06:00 UTC timing correction: repeated prefill pauses, not a cold-only delay

The saved 32K/C128 warmup took 95.101 s; measured bursts took 95.233, 94.975
and 95.112 s. This contradicts a one-off cold-start explanation. Earlier-half
requests (ranked by TTFT in each burst) had median stream rates 2.687-2.691
tok/s/user; later-half requests had 10.354-10.467. Current full-prefill
scheduling can pause already-started decode to admit later prompts. This is a
lead supported by source and the stream timing pattern, not a measured device
stall or a complete stage breakdown. Overall HTTP output rate is 172.26 tok/s,
including all prompt work for 32K-input/128-output requests. It must not replace
a decode-only long-context rate. Added warmup and per-burst timing logs for
future sweeps; the active immutable source was not edited.

## Oct 9, 06:14-06:18 UTC: immutable source preparation for head qualification

- Closed a release reproducibility gap: Git's tracked-file cleanliness check
  did not reject a locally qualified untracked or ignored precision policy.
  TTIS now checks each qualified runtime/policy hash against the advertised
  commit's actual blob. Seventeen tests passed. Re-preparing the native bundle
  against its real G0 receipt produced five byte-identical artifacts.
- Pushed TTIS `5f5302944` and separate model-source branch
  `anatarajan/qwen38-head-control-runtime-20261009` at `d3e8d6021f7`.
  The latter adds only the exact queued BFP8/HiFi2-head policy. All 28 source
  hashes match the frozen queue and the committed blobs; all 64 decoder
  policies are unchanged. No replacement G0 receipt or head score was created.
- Original hardware controller PID 917206 remains live in its original
  invocation; the 128K/C64 HTTP cell is active. Head-control PID 1103813 and
  post-head CPU/image PID 1209320 remain live and waiting. Serving workers
  continue prefill progress. No reset or source mutation was performed.

[Release source evidence](../galaxy-evidence/release-source-binding-v1/README.md).

## Oct 9, 06:21-06:34 UTC: conditional layer-by-layer HF accuracy diagnosis

- Added a B1 eager TP4 diagnostic using the queued CPU HF reference's exact
  teacher-forced tokens. It compares all 64 decoder inputs across four ranks,
  final normalization and logits, then runs the device head on the HF hidden
  state to separate head-only error. Checked the installed Transformers source:
  its last captured hidden state is after final norm, not the raw decoder output.
- Sixteen CPU tests passed locally and remotely; native runtime imports passed
  without opening hardware. Initial remote preflight found a fixture-interface
  mismatch hidden by the lightweight local fixture. Fixed the message-pattern
  argument and corrected that fixture; preserved the failed log and JUnit.
- Local SSH sandbox denied the initial staging command. After a successful
  approved retry, verified the partial new-directory contents before completing
  the immutable snapshot. Existing running/queued sources remained unchanged.
- At 06:33:25 UTC, launched qwen38-hf-layer-v1-20261009.service. PID 1255121 is
  live and waiting on the exact post-head CPU service. It runs only if full
  head-control GPQA remains below 177/198, and after CPU reference completion.
  A passing head control skips this extra hardware diagnosis. Its scope is
  numerical localization, not GPQA or throughput qualification.
- Original native HTTP sweep completed 128K/C64: 34.65 aggregate end-to-end
  output tok/s, median stream rate 1.348 tok/s/user, p50/p90 TTFT 139.74/222.84 s,
  median burst 236.39 s. These 128K-input/128-output bursts include all prefill
  work. The near-256K/C32 cell is now active; no full sweep pass is claimed yet.

[Diagnostic source and launch](../galaxy-evidence/hf-layer-reference-v1/README.md).

## Oct 9, 06:38-06:49 UTC: prefill scheduling review and state-slot prerequisite

- Confirmed the installed plugin pin already supports decode interleaving and
  scheduler token chunks. Qwen disables chunked prefill, so its internal model
  chunks bound memory without allowing the scheduler to insert decode between
  them. Smaller scheduling quanta need model qualification before enablement.
- Found and reproduced a plugin allocator defect for continuing prefills:
  changing host rows can change a request's state slot without moving its
  recurrent state. New arrivals can also take a continuation's slot. All four
  corrected regression cases fail on the original pin; 167 surrounding tests
  pass after preserving every live owner's slot. Hardware remains untested.
- One intermediate test assumed the wrong exact decode permutation. Replaced
  that assumption with the independently gathered physical-state locations,
  then reran both arms. An earlier command named a missing host test path and
  collected no tests; failed receipts are preserved.
- Upstream plugin main still contains the same allocator but has advanced to
  vLLM 0.29. Kept this fix on the deployed vLLM 0.26 plugin source in a separate
  worktree. Upstream push failed because this account has read-only access;
  published `e5b02d5` under `anatarajan/qwen38-chunked-state-slots-20261009`
  in `anatarajan-tt/vllm-tt-plugin`. Pre-commit passed before commit.
- This is a prerequisite for a future chunked-prefill experiment, not the
  cause of the current GPQA misses and not an enabled model/runtime change.
  Original near-256K HTTP work and all three followers remain active.

[Reproduction and remaining work](../galaxy-evidence/chunked-prefill-state-v1/README.md).

## Oct 9, 06:50-07:08 UTC: completed HTTP sweep and queued model continuation test

- Published the plugin-fix reproduction/evidence on the model branch at
  `d1c6ec926d7`; the plugin implementation remains separately pushed at `e5b02d5`.
  Verified all 652 indexed evidence paths were tracked and hash-matched.
- Native physical-Galaxy HTTP finished all nine admitted cells at 06:48:56 UTC.
  Near-256K/C32 measured 12.75 aggregate output tok/s including all prefills,
  1.105 median stream tok/s/user and 191.36/306.23-second p50/p90 TTFT. Three
  larger cells are KV-capacity guards, not failures or measured results.
  Collected the complete graph, CSV and compressed raw receipts.
- Added an opt-in 64-layer TP4 adapter diagnostic comparing eleven logit outputs
  between independent and interleaved requests with identical chunk boundaries.
  It exercises unaligned continuations, new arrivals, reordered rows, resident
  decode and physical state permutations while keeping request-owned KV pages.
  Host logits isolate this check; scheduler integration and final device RNG
  remain additional requirements. No capability or serving setting was enabled.
- Twelve new CPU tests passed locally. Native-host preflight passed 27 tests
  and six subtests; the hardware case was explicitly skipped. G0/source binding
  and the disabled chunking capability were also verified without a device.
- Initial staging SSH was sandbox-denied before connecting; the approved retry
  launched `qwen38-chunked-state-v1-20261009.service` at 07:05:06 UTC. PID
  1290750 and its invocation were observed live waiting on the exact HF-layer
  job. This preserves ordering after all existing hardware and CPU work.
  The new unit is bounded to 24 hours, 160 GiB and eight CPU cores.
- The native bandwidth follow-up continues making progress; one intermediate
  opposite-placement, 15-page, depth-4 case reached 278.14 GB/s/chip. This is
  not a final sweep winner or an attention/model uplift. Head-control G0/GPQA
  remains queued. No new accuracy or release-qualified configuration exists.

[HTTP sweep](../galaxy-evidence/native-http-sweep-v2/README.md),
[continuation diagnostic](../galaxy-evidence/chunked-prefill-hardware-v1/README.md).

### 07:10 UTC: delivery sweep completed; head control started automatically

The original controller exited successfully after all three stages at 07:06:04.
The preserved evaluation `passed=false` is not a hardware failure. Delivery
completed 54 correctness variants and 168 timed cases including controls. Best
delivered rates across the three sizes were 275.14, 278.14 and 279.84 GB/s/chip;
opposite placement with 15-tile packets and depth four/eight won. Read-only
controls remained 499.18-507.89 GB/s, with below 0.06% drift. This is still only
about 55% of measured read-only delivery, without attention math or page tables.
The mover remains unpromoted.

Observed head-control PID 1103813 in its original invocation, loading layer 33
of the first TP4 replica at 07:09. Its hardware phase began at 07:06:18, after
the exact predecessor exited; all later followers remain live and waiting.
No session connection is needed for that queue to continue.

[Complete delivery results](../galaxy-evidence/delivery-extended-v2/README.md).

## Oct 9, 07:13-07:22 UTC: agentic failure attribution while head G0 loads

- Reviewed all three naturally completed Tau3 failures, retaining 3/12 overall.
  Task 019's simulated user substituted a wrong ID despite the agent's correct
  tool handoff; task 051 contains an agent-invented verification time plus a
  stuck resubmission; task 102 failed verification/referral constraints. The
  initial theory of simulator-invented company age in task 102 was contradicted
  by the original instructions, which intentionally contain misleading facts.
  The local judge's rationale missed that distinction; DB scoring still failed.
- Added a read-only audit and invoked actual pinned upstream banking tools
  against a fresh synthetic in-memory database. Submission then denial leaves
  the original request pending and inserts a separate denial; resubmission
  reproduces the recorded duplicate-request error. No model calls, benchmark
  modifications or score repairs were made. This is not proof that correcting
  that blocker alone makes task 051 pass.
- Initial audit import lacked the task's PortAudio library path. The retry used
  the already installed task-owned path and completed; no native installation
  changed. Collected task/source hashes and diagnostic counts without exporting
  full conversations. Literal tool errors can have `error=false`; zero malformed
  JSON alone is insufficient evidence of tool correctness.
- Rechecked the experimental 6,010,501,632-byte OCI artifact on host disk and
  clean pushed TTIS packaging source `5f5302944`. Image runtime/hardware and
  SJC3 Helm qualification remain pending. Head-control PID 1103813 remains live
  loading replicas for G0; the most recent GPQA remains 170/198.

[Audit and reproduction](../galaxy-evidence/tau-review-v1/README.md).

## Oct 9, 07:31-07:39 UTC: authenticated Helm probe fix

- Checked whether no-device image validation could run early on the idle .34
  host. Its 28.07 GB available disk does not meet the existing 33.51 GB import
  budget including reserve. Compared the actual OCI configuration's layer
  prefix against all three existing Metal images: zero reusable prefix layers.
  Kept the original persistent .98 container-check queue; deleted nothing.
- Audited the generated runtime spec and rendered chart against the measured
  native launch. Found an authenticated-deployment failure: the inherited
  liveness route is `/v1/models`, but pinned vLLM protects that path and kubelet
  sends no bearer token. Executing the unmodified authentication class against
  synthetic ASGI requests reproduced 401, while `/health` passed the guard.
  This did not import vLLM, open devices, or start a listener. Source inspection
  confirms `/health` invokes engine health and reports a dead engine as 503.
- Changed only the Qwen overlay to select `/health` for all probes. Four chart
  cases failed before the fix; all 19 packaging tests passed after it, with
  API-key injection still enabled. Ruff and diff checks passed. Pushed TTIS
  commit `2485b039be071f75fa29adff6c84ecc87d60359e` under
  `anatarajan/qwen38-galaxy-release-20261009`.
- Regenerated the real native bundle from the exact G0 source pin. G0 receipt,
  runtime ModelSpec and image verifier remain byte-identical to the built image
  inputs. Preserved corrected values and a render receipt; the image is still
  unqualified and not registry-published. Also corrected the packaging report
  to include Tau3's step-limit outcome and partial manual-review findings.
- Head-control PID 1103813 remains live in invocation
  `ff89c12d6a9b4feaa268d7eff48b4132`, actively loading another replica for G0.
  No new accuracy result: full GPQA remains 170/198 and Tau3 remains 3/12.
  Review of recurrent-attention source did not establish another numerical bug;
  the queued HF layer comparison is still needed to localize the remaining gap.

[Helm probe evidence and corrected values](../galaxy-evidence/helm-health-v1/README.md).

## Oct 9, 07:43-08:02 UTC: head G0 passed; completion audit and image build queued

- Reviewed adapter sampling and recurrent-slot transitions. Existing seeded
  continuity tests cover remaps, intervening prefill and host/device sampling
  transitions. Source review did not establish a new cause of the GPQA gap.
- Found a reporting distinction: the harness can score a correct final-answer
  parser match before a length cutoff. All completed measured runs had zero
  such credits, so 170/198 remains unchanged. Added a separate completion score
  to the saved-response audit, preserving the original harness result and all
  198 denominator rows. The stricter qualification requires 177 naturally
  completed correct answers; it does not rewrite old answers or rewards.
- Twenty-two CPU tests pass, including rejection of a 177-point raw pass with
  one cutoff credit. Added a bounded read-only audit follower tied to the exact
  head service invocation. Source hashes match local and remote bytes; PID
  1345569, invocation `7172e4b27dbd47868376e12d70fa2815`, is live waiting.
  It opens no devices and leaves all existing frozen controllers unchanged.
- Head G0 passed at about 07:50:53 UTC: eight replicas, one passing JUnit test,
  44m20s, maximum concurrent/isolated TPOT ratio 1.000006. Same-shape short-
  context B1 timing changed from 286.27 to 281.33 aggregate decode tok/s;
  per-replica TPOT increased 1.61-1.76%. This is not a long-context cost claim.
- Collected the actual head G0 and prepared a new exact-source experimental
  bundle at model revision `d3e8d6021f7`. Serving startup followed automatically.
  Shared-memory wait warnings were followed by active warmup completions around
  07:59:48-07:59:54; they were not treated as proof of a hang or restart authority.
- Started the separate v6 head-image build on idle .34 at TTIS `2485b039b`.
  PID 3806410, invocation `95040bf8652248bca40bb52f13caeb76`, is active.
  Existing no-device bounds remain: four hours, 192 GiB, 24 CPUs, removable
  RAM build workspace. Added final host-disk preservation with complete source/
  destination checksum comparison and file/directory fsync, retaining 8 GiB.
- Staging first hit a sandbox network denial. A receipt-newline edit then
  produced a remote syntax error before any mutation; the corrected payload
  compiled and launched exactly one build. No existing job or native artifact
  was replaced. GPQA, image completion and container qualification remain open.

[Head G0 and bundle](../galaxy-evidence/head-g0-v1/README.md),
[completion audit](../galaxy-evidence/gpqa-completion-audit-v1/README.md),
[head image build](../galaxy-evidence/image-build-v6/README.md).

## Oct 9, 08:09 UTC: serving checks passed; head GPQA progressing

- Collected the API receipt: all six checks passed, including multi-turn chat
  and concurrent greedy repeatability at concurrency 128. This does not prove
  long-context capacity, model accuracy, or container/Helm qualification.
- Verified the exact head-control invocation is still active at PID 1103813.
  Full GPQA reports 116/198 complete, 110 correct and zero truncations at
  08:09:23 UTC. Fast-completing answers are not a representative final score;
  the unchanged requirement is 177 naturally completed correct answers.
- The completion audit is still waiting for the same invocation to finish.
  Tau3 has no newer result: 3/12, including six timeout/step-limit outcomes.
  The last completed native GPQA remains 170/198 with one incorrect cutoff.

## Oct 9, 08:12-08:31 UTC: head gate missed; image preserved and startup check queued

- Revalidated the exact head service and all queued followers as live. GPQA
  advanced from 137/148 correct to 166/194, with no truncations. Four questions
  remain, so the higher-precision head alone cannot reach 177/198. This does not
  establish the remaining numerical cause; the existing CPU/HF layer comparison
  remains the next diagnostic, and the full run is allowed to finish.
- The v6 build completed at 08:14:41 UTC in 15m25s. Both image stages passed
  source/import verification. The controller preserved the archive on local disk
  before exiting; independently reread its full 6,010,795,008 bytes and verified
  checksum, embedded manifest/config digests and exact source labels. Recorded
  the OCI digest and compressed build log. The image remains unqualified.
- Audited TTIS setup beyond imports and `--help`. Added a separate no-device
  startup probe that runs real wrapper setup and intercepts its final vLLM call.
  It compares the full declared argument/environment contract and the checkpoint
  symlink. Ten new regression tests plus 19 packaging tests pass; Ruff and diff
  checks pass. Pushed TTIS `54a5511fdcaed2dddc215c20f4ea95dad33dbdb6`.
- The first local test attempt referenced a removed old virtualenv and did not
  run; the existing Qwen tools environment ran all 29 tests successfully. Source
  staging and evidence collection initially hit sandbox SSH denial, then passed
  with approved retries. The startup job launched once, with frozen source and
  exact predecessor invocation. PID 1383768 was observed live, waiting.
- The new job checks the already queued native v5 image after CPU diagnostics,
  acquires the same device lock to avoid timing interference, and has no TT
  device/network exposure. It does not redirect frozen jobs, substitute the v6
  image or imply container model/accuracy/Helm qualification. No host data or
  unrelated image was deleted; .34 retained about 20.5 GiB free after preservation.

[Completed image evidence](../galaxy-evidence/image-build-v6/README.md),
[persistent startup probe](../galaxy-evidence/image-startup-probe-v1/README.md).

## Oct 9, 08:40-08:50 UTC: decoder controls prepared; sampled evaluation verified

- Confirmed the actual GPQA protocol and request implementation use temperature
  1.0, top-p 0.95, top-k 20, seed 42, thinking enabled and 65,536 output tokens.
  The separate performance workload's temperature zero is not GPQA's setting.
  Source inspection confirms the adapter supports the requested device-sampling
  parameters. The same seed across different numerical policies does not imply
  identical trajectories or isolate the cause of a score difference.
- Revalidated all six persistent controller invocations as live around 08:49.
  Head GPQA has 196/198 complete, 166 correct and zero cutoffs. Server PID
  1341860 still reports two generating requests and about 42 aggregate tok/s;
  an unchanged completion count was not treated as a hang or restart authority.
  Reference, layer, chunked-state and startup checks remain waiting.
- Re-read the completed Tau3 summary from the host: 3/12 remains the official
  pilot result. Existing simulator/tool findings do not justify changing it.
- Reviewed native recurrence and HF source for decay, beta, Q/K normalization,
  convolution history and gated norm semantics. This did not establish a new
  causal kernel bug. The queued matched-input HF comparison is still required.
- Prepared explicit BFP4/HiFi2 and BFP8/HiFi2 decoder configurations with the
  existing BFP8/HiFi2 head. Validated the real policy loader and all 64 layer
  mappings and saved exact differences/fingerprints. No serving default or
  frozen remote controller changed; neither control has been queued or run.
- Read checkpoint configuration and relevant safetensor headers without loading
  weights. Recorded source hashes and calculated a static per-chip estimate
  including both interleaved and DRAM-sharded projection representations. BFP8
  raises these known resident allocations from 17.669 to 23.477 GiB/chip, leaving
  8.398 GiB versus the SoC bank map before excluded allocations. Workspace,
  temporary buffers and allocator overhead must still be measured; this does
  not prove admission or predict a performance/accuracy improvement.
- Initial live-status SSH was denied by the sandbox and succeeded on the
  approved retry. One source search used nonexistent weights.py/linear.py names;
  the allocation definitions are in decoder.py and model.py and were inspected.

[Prepared controls and evidence](../galaxy-evidence/decoder-precision-controls-v1/README.md).

## Oct 9, 08:53-08:54 UTC: precision controls pushed; Tau3 timeout evidence extended

- Published decoder controls and capacity assumptions at `92dcd68096e` on the
  existing model branch. Pre-commit passed; all 722 indexed tracked artifacts
  matched their sizes/hashes. Staging first encountered the protected worktree
  Git index; the approved retry succeeded. No remote run was changed.
- Extended the read-only Tau3 review with per-role client-call timing from the
  saved requests. In the four 1,200-second task timeouts, completed agent calls
  account for 1,034-1,149 seconds; maximum prompts are 51,402-120,851 tokens.
  These wall times include prompt work and other serving overhead, so they are
  not pure decode rates. Unrecorded in-flight calls are not assigned outcomes.
- Reviewed final saved actions for the incomplete trials. At least one simulator
  response disregards the agent's waiting-period explanation; another task had
  successfully executed credit tools and was filing reports before timeout.
  Neither observation proves eventual task success. Apparent nested JSON
  escaping in a raw response was checked against successful downstream tool
  responses and was not labeled a parser bug. Pilot score remains 3/12.
- Initial trace inspection printed too much nested reasoning metadata; the
  follow-up artifact contains only hashes, counts, token totals and durations.
  Raw conversations remain on the allocated host.

[Extended Tau3 review](../galaxy-evidence/tau-review-v1/README.md).

## Oct 9, 08:53-08:56 UTC: head GPQA completed; persistent follow-up advanced

- Full head-control GPQA finished at 08:53:32 UTC: 166/198 (83.84%), 51m43s,
  one incorrect output-budget cutoff. All 198 raw-response hashes match the
  scored receipts. The completion audit confirms 166 naturally completed
  correct answers, with 31 natural-stop failures. No qualification gate changed.
- Compared identical-input/protocol receipts: 157 both correct, 13 native-only,
  nine head-only, 19 neither. The previous native result is 170/198; the head
  change did not qualify. The single sampled comparison does not prove lower
  expected accuracy or identify a causal kernel defect.
- Confirmed the head service is terminal and the original post-CPU controller
  advanced at 08:54:05 UTC. It is importing/checking the native v5 image, then
  will run its frozen CPU HF reference. Remaining hardware followers still wait
  for that reference. There was no restart, source replacement or queue change.
- The first attempt to collect JSON through a tool response exceeded its
  output limit and failed parsing before writing artifacts. A bounded local
  collector captured SSH stdout directly instead and saved exact receipts plus
  a deterministic gzip of the full audit. No raw private answer text was copied.

[Completed head-control evidence](../galaxy-evidence/head-gpqa-v1/README.md).

## Oct 9, 08:58-09:15 UTC: repaired diagnostics; decoder error localized and controls launched

- The original post-CPU queue completed with two failed stages. Image transfer,
  checksum and import succeeded, but Docker exposes the OCI manifest digest as
  its image ID on this host. The checker incorrectly inspected the config digest.
  Read-only inspection by the manifest digest confirms the descriptor/source
  labels. Runtime/startup checks still need recovery; no image was deleted.
- The HF reference loaded the checkpoint, then failed serializing empty sets in
  Transformers loading metadata. The exception handler hit the same error and
  left a stale loading report. Added deterministic set serialization and atomic
  report writes, with three regression tests. The first isolated local test
  fixture used pytest.raises with an incompatible positional signature; fixing
  that temporary fixture produced three passes. Native preflight later passed
  all 19 report/reference/follower tests.
- Old followers stopped as designed after failed dependencies; systemd MainPID
  was zero and terminal receipts were preserved. Launched a fresh CPU-only-model
  reference and follower in v2 directories. CPU model inference finished eight
  positions in about 34 seconds plus import/cleanup. Its host TTNN quantization
  conversion nevertheless initialized UMD and discovered/started devices; the
  saved log proves that hardware access despite the frozen legacy
  `hardware_opened=false` field. Documented this explicitly. The device lock was
  held throughout, and all model parameters/forwards were on CPU.
- The reference loaded without missing, unexpected or mismatched keys. Head
  BFP4/BFP8 weight-only RMS errors were 8.148%/0.532%, both matching all eight
  top tokens. These short-prompt values are not GPQA score predictions.
- The exact layer follower invocation `60840a4a755740689518e67aa1afc4e8`
  completed all eight hardware steps and clean shutdown. Full logits differ
  from HF by 20.1-70.3% RMS versus 0.60-0.83% for the device head on HF inputs.
  All top-1 tokens agree. Initial prefill error accumulates through the decoder,
  motivating a fidelity/weight split but not proving a kernel defect.
- Source inspection confirms the installed HF full-attention output gate is
  sigmoid, matching this implementation. The plan's swish label is not a reason
  to change that operator. No causal recurrence/gating bug was established.
- Added explicit unqualified decoder-control support to the diagnostic. G0
  still verifies the original model source, mesh and baseline precision; control
  validation permits only uniform BFP4/BFP8 decoder weights and HiFi2 while
  preserving the head, state, KV and attention policy. Runtime receipts record
  the actual candidate policy and deny inherited serving qualification.
- The first control job stopped at import due to a missing helper module in
  its frozen source snapshot. A first collector exposed the missing queue file;
  the controller traceback and failed MainPID-zero service established startup
  failure before hardware. Added the helper and its tests plus explicit import
  preflight. The corrected bundle passed 34 tests with one hardware skip.
- Started `qwen38-hf-controls-v2-20261009.service` at 09:14:25 UTC, observed PID
  1433402 and invocation `79a488c86b8a4304b747388152292642`. It runs BFP4/HiFi2
  then BFP8/HiFi2 sequentially under the shared device lock, stopping on failure
  or unproven cleanup. Two-hour outer bound, 160 GiB, eight CPU quota. This is a
  persistent numerical comparison; neither a new GPQA result nor a performance
  claim. The original chunked-state and image-startup failures remain preserved
  for recovery after the current accuracy priority.

[Recovered reference and failures](../galaxy-evidence/hf-reference-recovery-v2/README.md),
[persistent decoder controls](../galaxy-evidence/hf-decoder-controls-v2/README.md).

- Publication inventory validation caught three collected `.log` files excluded
  by the repository ignore rules after the initial staging. They were preserved
  as deterministic gzip files and the index corrected before pushing. The first
  local commit was made before that failed validation was inspected; the
  follow-up commit repairs the publication inventory without rewriting history.

## Oct 9, 09:25-09:31 UTC: BFP8 improves reference agreement; full GPQA launched

- Both decoder controls completed and cleaned up. BFP4/HiFi2 produced identical
  saved full-logit metrics to BFP4/LoFi; BFP8/HiFi2 reduced mean relative RMS
  from 38.65% to 9.16% across eight teacher-forced positions. All top-1 tokens
  agree under both policies. The result motivates a full benchmark without
  asserting that the short probe predicts GPQA or proves all prior causes.
- Added a tested decoder-control predecessor gate to the existing persistent
  accuracy follower. Fourteen focused local tests passed. Native preflight of
  the frozen bundle passed 423 tests and 40 subtests, with one expected skip.
- Launched BFP8 eight-replica G0 then unchanged full 198-question GPQA at
  09:28:02 UTC, invocation `ecb6408f79414ed0974c6352f006bce5`, PID 1449553.
  Precision, source, sampling, output budget, full denominator and 177/198
  target are frozen. Host-memory bound is 256 GiB, total runtime bound eight
  hours. No precision/performance promotion was made.
- A first read-only collection printed too much comparison JSON and truncated
  its display. A local collector captured full raw bytes instead, preserving
  both comparisons, final queue, logs, XML, launch and source-manifest hashes.
- First completion-auditor staging failed because its helper is absent from
  the frozen inference snapshot. No directory, service or hardware was changed
  by that failed attempt. Staged a separate audit-only bundle and passed all
  22 tests. Its persistent service waits for the exact new GPQA invocation and
  will verify all 198 response hashes and count incomplete answers as failures.

[Completed decoder controls](../galaxy-evidence/hf-decoder-controls-v2/README.md),
[full BFP8 qualification launch](../galaxy-evidence/decoder-gpqa-v1/README.md).

## Oct 9, 09:35-09:37 UTC: restore latency follow-up behind accuracy

- Pushed full GPQA launch, decoder-control evidence and follower changes at
  `240b4c24d6b`; all 783 indexed tracked artifacts matched their hashes and
  pre-commit passed. The active BFP8 invocation remains unchanged, with its
  first physical TP4 replica loaded in 306.66 seconds and later replicas loading.
- Restored the earlier chunked-state diagnostic in fresh v2 source/control/result
  directories. It waits for this GPQA's exact completion auditor, then acquires
  the shared device lock. The prior failed queue remains preserved.
- Verified the original native BFP4 runtime against its G0 receipt and kept
  serving chunked-prefill support disabled. Preflight passed 27 tests and six
  subtests with one hardware skip. The test covers eleven host-logit comparisons
  under interleaving and physical slot movement; it does not qualify the plugin
  scheduler or device sampler and does not imply a Tau3 score improvement.
- A read-only source search referenced a nonexistent run_galaxy_replicas.py;
  the actual eight-replica execution is in tests/test_galaxy_replicas.py. It loads
  all eight models serially, so one replica's five-minute load is not the total
  G0 loading time. No running job was restarted or changed.

[Restored state test](../galaxy-evidence/chunked-prefill-hardware-v2/README.md).

## Oct 9, 09:39-09:46 UTC: repair imported-image check while GPQA G0 loads

- Revalidated the original BFP8 hardware controller and exact invocation as live;
  two replicas were loaded initially, three at the last check. No hardware job
  was interrupted. The previous goal turn made progress by completing numerical
  comparisons, adding the guarded qualification runner and launching durable
  full GPQA plus completion/state follow-ups.
- Added a TTIS operator validator for Docker's config-ID and manifest-ID image
  stores. It preserves both immutable pins and checks OCI metadata content,
  linkage, platform, ordered root filesystem and runtime configuration. Eleven
  tests passed locally and natively, including modern/legacy identities and
  rejection of changed configs, layers, platforms or corrupt metadata.
- Live check on the already imported native v5 image passed all 46 layers and
  runtime metadata. Only metadata was read; the archived payload is not rehashed
  by this checker. It does not open hardware, run containers or confer accuracy.
  Pushed TTIS code/docs at `52390d1b0d4015fbde230fbcfb7e357b3b84e24c`.
- Restored container source/import, entrypoint-help and TTIS startup-handoff
  probes in fresh v2 directories, behind the exact queued chunked-state job.
  The sequence repeats immutable identity validation under the coordination
  lock. It uses networkless containers without devices, read-only roots and
  bounded task-owned tmpfs. The new service is live and waiting; no startup
  pass is claimed. Old failed import/startup receipts remain unchanged.
- Routine local searches encountered absent AGENTS/.agents/deploy paths and a
  shell unmatched release-document glob; scoped repository search located
  scripts/release/QWEN38_GALAXY.md. A context-only patch failed before edits;
  the corrected anchored patch added the validated import instructions.

[Image identity result and queued recovery](../galaxy-evidence/image-recovery-v2/README.md).

## Oct 9, 09:49-10:09 UTC: quantify precision tradeoff; queue matched performance

- The user asked how much BFP8 costs. Rechecked the exact active qualification:
  four replicas loaded at the first check, seven at collection, no timed decode
  measurements yet. Explained that loading and numerical-error results cannot
  be used as inference throughput measurements.
- Derived an explicitly unmeasured bandwidth model from the recorded padded
  matrix geometry. Streamed weights rise from 3.925 to 7.113 GB/chip/step.
  At B16 the traffic-only TPS reductions are 30.1% at 16K and 24.7% at 32K.
  The calculation omits HiFi2 compute cost and prefill, so these are not actual
  engine predictions. The estimate retains its inputs, hash and assumptions.
- Launched a persistent three-policy, two-context comparison after the exact
  queued chunked-state job. It shares the coordination lock with image checks.
  Uses frozen active-qualification source, B16 on one TP4, 128 output tokens,
  one warmup and three measurements. BFP4/LoFi -> BFP4/HiFi2 -> BFP8/HiFi2
  separates fidelity from weight precision while keeping head/KV/state fixed.
- Eighteen native tests and frozen-import/plan validation passed. Observed
  PID 1491803, invocation `122b1077bed2438d9802360ab6277514`; source-manifest
  hash matches the current accuracy bundle. The controller stops on failure
  or unproven cleanup. This does not promote any policy or bypass GPQA.
- Read the earlier completed head G0 timing only as a baseline: aggregate
  281.33 decode tok/s with one short-context user per replica. It is not
  comparable to a B16/32K measurement, and the new BFP8 G0 has no timing yet.

[Precision cost estimate and persistent comparison](../galaxy-evidence/precision-perf-v1/README.md).

## Oct 9, 10:13-10:21 UTC: BFP8 G0 passes; exact-SHA packaging build launched

- BFP8 G0 passed all eight physical TP4 replicas, including matched greedy
  repetitions and concurrency interference checks. It finished at 10:13:29
  UTC after 45m19s including serial model loads. Full traced aggregate decode
  was 211.54 tok/s versus 281.33 for the head-only control: 24.81% lower at
  one short-context user per replica. Source hashes match except for precision.
- The unchanged persistent controller advanced to serving. At 10:19:15 workers
  were actively loading layers 54-56, with zero GPQA results and no reported
  admission error. Replied to the user's score question without treating G0
  or the short reference comparison as a GPQA pass.
- Created a separate runtime worktree from head-control commit d3e8d6021f7,
  added the two already tested decoder precision configs, and verified all 33
  runtime/config files against the frozen active source. Pushed exact runtime
  `20619e008a236aaf393937b222a60a5b03e49cdc` on the anatarajan BFP8 branch.
- Prepared its exact-G0 TTIS runtime/Helm bundle and extended existing wrapper
  parameterization to the all-BFP8 policy. The combined packaging/startup/OCI
  suite passed 44 tests. Pushed TTIS `bbb1ca07e7cd0f97af47745a1686545a16089b9c`.
- Verified .34 had no running build containers, 22.04 GB disk and 291.91 GB
  tmpfs free. Started bounded persistent image build v7 there, preserving .98
  for hardware evaluation. It was observed building as PID 3977785, invocation
  `ea312e233d9e4dd5995f81ccadd7f564`. Image digest and accuracy remain unqualified.
- Local searches for old temporary release Python environments found them
  absent; the existing qwen38-tools environment successfully imported the
  preparation CLI and ran the 44 tests. No dependencies were installed.

[BFP8 G0 result](../galaxy-evidence/decoder-g0-v1/README.md),
[exact-SHA build](../galaxy-evidence/image-build-v7/README.md).


## Oct 9, 10:29-10:43 UTC: BFP8 GPQA advances; image completes and checks queued

- Revalidated the exact live GPQA process rather than inferring progress from
  its queue file. Early completion-order snapshots advanced from 43/46 to
  156/169 and then 162/180 correct, with no truncations. The unchanged full
  198-question gate and independent completion audit remain pending. All six
  API checks passed, including multi-turn and concurrent repeatability.
- Image v7 completed on .34 at 10:33:37 UTC (14m58s including preservation).
  Source/import checks passed in the built runtime. Its 6,011,314,688-byte OCI
  archive was checksummed on both reads, fsynced and preserved on host disk.
  Manifest digest is 79f7b4469a6ec2bcce5204399b37b2aeced8be7f260dd98f7007ad41f8813055.
  The builder stopped, and the systemd service completed successfully.
- Queued an image-specific import/startup controller after the exact precision
  comparison, then the common lock. It waits before copying to avoid competing
  with hardware work. It verifies the pinned TLS archive, byte count, checksum,
  Docker identity, runtime imports, entrypoint and real TTIS handoff, without
  TT devices or inference. PID 1582084, invocation
  c3068a78ac924fd3a1de1c847be0e53c; observed live and waiting.
- A fresh short-lived TLS server on .34 is restricted to the one digest and
  source .98. Its source, public certificate, initial state and launch evidence
  are retained; the private key stays on the host. Preflight verified source
  hashes, syntax, the helper CLI and an authenticated HEAD with exact length.
  Shared helper tests remain 44 passed from the unchanged source; no redundant
  CPU suite or container pass was invented.
- The first stage guard rejected a successful transient build unit because
  systemd had already collected it. It stopped before transfer files were
  created. Read-only inspection proved its completed artifact and stopped
  builder; the corrected guard accepts that terminal case and still rejects
  replaced or running invocations. No active experiment was restarted.
- Reviewed remaining Tau3 integration: the queued native BFP4 state diagnostic
  is not a full plugin/sampling qualification. Chunking stays disabled until
  that diagnostic and an actual plugin continuation test pass. The installed
  plugin remains unchanged; its isolated continuation-slot fix is already
  pushed. No extra Tau3 rerun or chunked deployment has been launched yet.
- A broad temporary-file name search encountered an inaccessible daemon
  directory; all relevant task scripts were found. Subsequent reads were scoped.
  A JSON evidence-index search was overly verbose but made no changes.

[Serving evidence](../galaxy-evidence/decoder-gpqa-progress-v1/README.md),
[completed build](../galaxy-evidence/image-build-v7/README.md),
[image check queue](../galaxy-evidence/image-startup-bfp8-v7/README.md).


## Oct 9, 10:46-10:51 UTC: GPQA gate unreachable; preserve full run

- At 166/188 the maximum possible BFP8 score became 176/198, below 177/198.
  The exact original process remained active; a later check found 168/192
  with zero cutoffs. Reported this without claiming a final score, stopping
  the run, changing the gate, or selecting a different seed.
- Prepared diagnostic-only BFP8/HiFi4 and BF16/HiFi4 decoder configurations,
  preserving the current BFP8/HiFi2 head, FP32 state, BFP8 KV and native
  recurrence. Extended only the diagnostic validator; 25 local host tests
  passed, including rejecting mixed projection modes and changed head/state.
  These are not queued, measured or deployed. The user's instruction to let
  the current full run finish and capture its result remains in effect.
- The host tests exclude device conftest imports and reuse its expect_error
  fixture, retained in the evidence. Pytest's cache write under /dev failed
  harmlessly; all assertions and the saved JUnit report completed. No remote
  CPU validation or hardware operation competed with the active evaluation.
- First publication pre-commit adjusted EOFs on two raw transport receipts.
  Preserved their original staged bytes as gzip instead, retaining exact
  evidence. A guessed host test-fixture path was absent; the actual fixture
  was located in repository-root conftest.py.

[Prepared reference controls](../galaxy-evidence/higher-precision-diagnostic-plan-v1/README.md).


## Oct 9, 10:53-11:00 UTC: preserve full run and automate final capture

- Classified the preceding turn as progress: completed BFP8 build evidence,
  a persistent image-check follower, tested unqueued diagnostic policies and
  publication at 8807adce8fb. The original evaluation was still verified live.
- Honored the user's instruction to let the full run finish. Read the exact
  service and recent server counters: the five remaining requests continued
  generating at about 19-24 tok/s/user, with no waiting queue. Later progress
  reached 170/196 and two outstanding, zero truncations. No restart, reset,
  serving change, new inference request or new precision run was performed.
- Added a persistent read-only final capture after the exact evaluation and
  independent audit. It requires terminal services, clean owned-worker shutdown,
  all 198 scored receipts, matching audit hashes and the same comparison input
  and sampling protocol. It retains scores, cutoff audit, matched prior outcomes
  and client timing distributions with explicit measurement boundaries.
- Launch/preflight verified the current original invocation and state. The
  capture service was observed alive as PID 1601319, invocation
  75fb7d86dbbe47d9ad4af9985bac6014, waiting without opening devices. Limits are
  one CPU, 512 MiB, nine-hour wait and ten-hour service lifetime; it survives
  session disconnection. No remote helper/source from the original run changed.
- Validated the terminal reader against the completed head-only control by
  changing only path/identity constants in a separate in-memory copy. It
  reproduced 166/198 and validated all ten generated JSON/gzip artifacts,
  including source/result hashes and matching baseline protocol. This does not
  create or modify an eval score, and is not the live BFP8 result.

[Persistent result capture and validation](../galaxy-evidence/decoder-gpqa-final-capture-v1/README.md).


- Publication correction, 11:05 UTC: a temporary index-refresh helper reused
  its destination variable inside a file loop, writing the inventory JSON over
  the new compressed validation-script copy instead of updating the index.
  The hash check caught the stale logbook entry, but the orchestration failed
  to stop the subsequent commit/push. Restored the exact compressed source,
  corrected the inventory destination, and verified every tracked hash and
  gzip payload before a separate correction commit. No host source, running
  service, benchmark receipt or score was affected.

## Oct 9, 11:09-11:32 UTC: benchmark configuration audit finds corrupt GPQA choices

- Read-only comparison against the official pinned model card and checkpoint
  confirms eight config/tokenizer/index artifacts byte-for-byte, xhigh thinking,
  temperature 1/top-p .95/top-k 20, and exact live tokenization for a synthetic
  prompt. Installed harness revision matches its pin. Model-card GPQA protocol
  equivalence remains unproven; no sampling or gate changes were made.
- Found a concrete harness defect: its GPQA preprocessor removes every
  square-bracketed span. This corrupts 38 answer choices in 12 questions; seven
  were incorrect in the BFP8 run. Row 171 loses all four distinctions and the
  first-string-match label no longer tracks the original correct entry.
- Added a benchmark-only preserving preprocessor with tagged labels and an
  explicit protocol version. All 28 CPU regression tests pass. Actual dataset
  preparation verifies all 198 answer texts and unchanged shuffle positions;
  row 171's label is corrected. Fifteen prompts change (12 bracketed, three
  internal-whitespace-only), with a new recorded input hash. No fresh inference
  or post-hoc score modification was performed.
- Audit development issues: first isolated test staging lacked models.perf
  (27 pass, one import failure). Second staging passed 28 tests but its data
  diagnostic wrongly required the collision-derived label to be preserved.
  Final staging verifies permutation preservation and permits only proven
  collision label correction. Earlier scratch normalization and BatchEncoding
  comparison errors were corrected, not attributed to serving.
- The untouched original run finished 11:26:26 UTC at 171/198 (86.36%), no
  truncations, 59m51s, below 177. Mean client TSU 12.93 and aggregate output
  340.45 tok/s. Persistent completion audit and capture completed; copied
  final receipts, comparisons and logs locally without private question text.
- The original score remains recorded under the defective protocol. A fresh
  full corrected run is required and has not been launched by this audit.
  Existing persistent follow-up jobs were not modified. Evidence and detailed
  findings: galaxy-evidence/gpqa-config-audit-v1/ and decoder-gpqa-result-v1/.
- Pre-commit required the repository expect_error fixture in the new negative
  tests. Adopted it and repeated all 28 CPU tests and dataset preparation; final
  receipts are under gpqa-config-audit-v1/final-validation/.

## Oct 9, 16:14-16:43 UTC: GPQA-first bounded queue and release-process review

- Queued corrected full GPQA first, unmodified pinned OpenBench GPQA second
  (explicit one-epoch/64K-output bounds), then native BFP8 batch/context and
  prefill-token-budget sweeps. Eight-hour persistent unit, process-group cleanup,
  shared device lock, frozen sources, separate scores; no best-of selection.
- 47 CPU checks pass. OpenBench mirror exactly matches the original 198 records;
  synthetic HTTP probe confirms request parameters and final-answer-only scoring.
- v1 failed before inference with a DP5 fabric topology mapping error. Preserved
  failure, verified owned-worker shutdown and launched v2 with locked Galaxy reset.
  v2 entered corrected GPQA at 16:37:22 UTC; 48 results observed by 16:43 UTC.
  Failure recovery is demonstrated; the initiating hardware/software cause is not.
- Collected completed overnight precision rates and chunked-state success. BFP8
  costs 8.52%/7.93% decode TPS versus BFP4/HiFi2 at 16K/32K, B16. These are
  measured native TP4 results, not eight-replica HTTP measurements.
- Read the provided Models CI release-process PDF and recorded candidate pins,
  dev-only catalogue policy, Shield On-dispatch/nightly/release prerequisites,
  stable/main fix propagation and automated production promotion. Existing
  branches/image do not yet qualify a coordinated release. No release mutation.
- Evidence: galaxy-evidence/gpqa-first-v2/. Frozen published source ASTs match
  the live copy; formatting changes were not applied to running snapshots.

## Oct 9, 17:09 UTC: GPQA accepted; release work scoped

- User explicitly accepted GPQA as close enough and asked for remaining release
  work. Live progress: 174 correct among 194/198 completed, zero truncations.
  Preserve the remaining generations and final score; do not rewrite the original
  threshold or claim its strict gate passed. GPQA is no longer a user acceptance
  blocker. The persistent queue continues unchanged.
- Inspected the image probe stderr: exit 2 is argparse rejecting a missing
  `--tt-device` argument in the probe's wrapper invocation, before hardware or
  serving. Runtime imports and entrypoint help had passed. Fixing the probe and
  proving real container inference remain separate steps.
- Inspected retained Tau3 evidence: 3/12 overall, four task timeouts, one
  infrastructure error, zero malformed tool calls among 277 tool calls. This
  older pilot does not qualify current BFP8 agentic performance.
- Remaining release path: container/API/tool validation; publish SHA/digest-pinned
  artifacts and verify Helm deployment; Galaxy dev catalogue and Shield jobs;
  On-dispatch/nightly/release CI, stable alignment and release-team promotion.

## Oct 9, 17:20-19:09 UTC: final GPQA, image handoff and OpenBench timeout fix

- Corrected full GPQA completed 176/198 (88.89%), no truncations, 50m49s.
  User acceptance is retained separately from the original strict 177 gate,
  which remains missed in the untouched receipt. Client mean decode TSU was
  13.01 and aggregate output 419.48 tok/s on this variable-length evaluation.
- Real BFP8-image startup handoff passed after the external probe supplied
  `--model` and `--tt-device` to TTIS. This closes wrapper handoff only, not
  actual hardware inference. Image digest and runtime bytes are unchanged.
- Separate pinned OpenBench finished 170/190 scored, eight request errors;
  170/198 (85.86%) including errors. All errors were HTTP read timeouts after
  1,802-1,806 seconds. The SDK retained 600-second timeout/two retries despite
  Inspect's separate 7,000-second/no-retry generation policy. Original protocol
  claimed no retries but did not enforce that at the HTTP layer; preserve it.
- Set timeout and zero retries on the OpenAI-compatible SDK explicitly. A real
  delayed-response HTTP regression in the pinned evaluator proves one request
  on timeout and successful scoring with a longer deadline. Initial diagnostic
  v2/v3 probes inspected timeout representation incorrectly; v4 passed.
- Added a bounded TTIS container/API supervisor; 43 targeted CPU checks pass.
  Replaced its verified waiting-only v1 job with v2, which runs a fresh full
  OpenBench evaluation after API/tool checks against the resident container.
  PID 2335933, invocation d369052ee99040c1810ae8d56a76dd51 confirmed active.
  The current GPQA/perf controller PID 1982991 remained active and unchanged.
- The image job waits for the exact predecessor and clean completion, then
  acquires the device lock. It survives disconnect, uses removable host-local
  caches and has bounded cleanup. No shared CI, registry or Helm deployment
  was modified. Evidence: galaxy-evidence/gpqa-first-final-v2/ and TTIS branch
  scripts/release/evidence/qwen38-20261009/.

- Publication follow-up: restored copied status/JUnit metadata omitted by a
  hash-only evidence-index refresh. Raw receipts, scores and running jobs were
  unchanged. Revalidated all hashes and preserved every original metadata field.

## Oct 09, 19:49 UTC: BFP8 optimized recurrence queue and independent Tau preparation

- User reiterated 20 TSU at B16/B32 and requested persistent queued runs.
  Current native BFP8 32K B16/B32 is 85.11/135.99 ms; remaining graph/GDN
  optimization is required. Old BFP4 component profiles are not a complete
  current BFP8 critical-path attribution.
- Added a fixed-precision shared-QK BFP8 policy and immutable follower. It waits
  for the exact live container/API/OpenBench invocation and successful cleanup,
  then measures native/candidate/native at 32K/16K B16/B32, profiles four 32K
  layer cases, and runs fresh eight-replica G0/full corrected GPQA. Same lock,
  bounded waits/stages, no automatic promotion, survives session disconnect.
- Host CPU preflight: 456 passed, one skipped, 40 subtests. The actual repo
  fixture caught no errors after repairing the Tau tests' missing message args.
  Live follower PID2479417, invocation ebc4580116294d7f803953362980a502.
- Original queue remains live, now at bfp8-budget16k. Its native long-context
  sweep completed cleanly: 128K B4/B8 17.623/13.644 TSU, near256K B4 14.454 TSU.
- Prepared official pinned AWS Tau-Verified airline, all50 tasks, using independent
  OpenRouter GPT5.1 simulator and GPT4o-mini judge. CPU task/tool preflight and
  actual LiteLLM synthetic transport passed, including timeout/no-retry behavior.
  No OpenRouter credential is configured and no scored Tau evaluation is queued.
- First staging/capture SSH attempts were sandbox-rejected before remote action;
  authorized retries succeeded. No weights/native install/firmware/NFS changed.
  Evidence: galaxy-evidence/bfp8-gdn-followup-v1/ and tau-verified-preparation-v1/.

## Oct 9, 20:59-21:56 UTC: capacity recovery and fused GDN epilogue

- Original perf queue completed the 16K budget sweep, then the optional 64K
  budget ran out of contiguous device DRAM at 32K/B32: 71.3 MB required per bank,
  63.4 MB largest free block. Clean device closure is recorded. Dependent
  container and GDN jobs stopped before hardware access, retaining their failures.
- Added a narrow clean-allocator-limit outcome for that optional experiment and
  an explicit recovery audit of exact stopped invocation, exit code, preceding
  stages, raw OOM receipt and lock availability. 35 targeted tests passed.
  Started image-hardware-v3 and bfp8-gdn-v2, preserving source snapshots and all
  failed evidence. Container reset succeeded; eight workers are loading.
- 16K prefill budget worsened matched 32K/B32 input TPS (4692 to3714), with no
  useful decode change. 64K improves B16 prefill (5321 to5870) but cannot admit
  B32 in this run. Keep 32K as the working budget; missing cells remain not_run.
- Implemented standalone GDN epilogue consuming raw FP32 recurrence rows,
  combining output tiling, gated RMSNorm and z multiplication in one program.
  No default/model path changed and no speedup claimed.
- Simulator v1 failed in the test reference: explicit deallocation of padded
  gate freed an alias of the candidate gate. Retain reference intermediates.
  v2 executed but exceeded the unchanged 0.1% per-head RMS gate (0.28624%).
  v3 isolated a bit-identical norm and different final multiply. v4 preserves
  the BF16 intermediate and uses native-style SFPU multiply plus explicit BF16
  RNE. All nine B1/B16/B32 allocation/rebinding cases now match bit-for-bit;
  unchanged inputs and zero padding also pass. Hardware/eval gates remain open.
- Preserved failed and passing simulator logs, manifests and exact compressed
  source for all four attempts. CPU simulator units use separate cache, one
  CPU,8GiB/45min bounds; no physical devices. Some sandbox-blocked SSH attempts
  were retried after escalation; interrupted initial launch was verified absent
  before starting it. Current physical runs were not restarted on poll timeout.

## Oct 9, 22:00-22:08 UTC: physical epilogue follower

- Verified recovered image-hardware-v3 and bfp8-gdn-v2 still active. Container
  progressed through layer 63 and workers started reporting model loaded.
- Added a physical TP4 epilogue test: B32/B16/B8/B1, DRAM and interleaved L1,
  all-rank native intermediate/final checks, zero padding, input preservation,
  independent allocation rebinding and changed-input trace replay.
- Matched native/fused/native trace brackets use five samples of 100 replays.
  Native drift above 3% withholds a speedup claim; linear 48-layer savings are
  labeled projections, not measured full-model improvements.
- Added and tested an exact-predecessor/clean-receipt gate. The new persistent
  unit waits behind BFP8 performance/profiles/full GPQA, then uses the existing
  hardware lock. No model path or precision policy was changed.
- Frozen-source preflight: 450 tests and 40 subtests passed, one skipped;
  physical pytest collection passed. This inherits the older frozen BFP8 source
  plus new epilogue files, so its test count differs from the local latest tree.
- Launched qwen38-gdn-epilogue-hardware-v1-20261009.service, invocation
  16c50a2f3e4e428fbefe4c87c662cacb, waiting on exact BFP8 v2 invocation.
  Host limits 32 GiB and 8 CPUs; hardware bound 90 minutes; no reboot resume.
  Stored launch, source manifest, preflight and three-service status snapshots.
- Initial staging/collection helper SSH attempts hit sandbox network denial;
  retried with escalation. Pre-commit checks passed before freezing source.

## Oct 9, 22:10-22:24 UTC: prioritize measurements and direct-input prototype

- Confirmed current BFP8 full-model baseline from sweep receipts: at32K,
  B16=11.749 and B32=7.354 output tok/s/user (188/235 aggregate per TP4).
  No newer optimized BFP8 result was available;20TSU remains unachieved.
- Implemented a standalone direct tiled-input preparation kernel, reusing
  existing FP32 Q/K normalization and retaining external native exp. It bypasses
  adapter layout/cast/packing work without changing precision or model defaults.
- Simulator v1 exited on unsupported SETDVALID while executing the first case,
  before any completed comparison. Preserved failed process/probe/log/source
  evidence. The probe's stale running state is not treated as live or passing.
  No new preparation implementation was inserted into hardware qualification.
- After the user requested faster progress, prepared and CPU-tested an explicit
  performance-first order (457 tests,40 subtests passed, one skipped). A separate
  persistent reorder verified exact owned invocations/container label, stopped
  the waiting followers and container, verified removal and lock availability,
  and marked the device for a locked reset. Existing receipts remain canceled.
- Launched perf-priority-v1: B32 profiles before B16, then the previously
  simulator-tested epilogue, matched full-model sweeps and optimized full GPQA.
  Requeued container/API/OpenBench as image-hardware-v4 afterward, retaining its
  image/source/checkpoint pins and existing v3 cache. No NFS/firmware mutation.
- First native32K/B32 profile completed and passed strict test/cleanup/all-rank
  timing collection. Wrapper warned about CSV location, but the collector found
  the single CSV under the configured Tracy output path. Shared-QK B32 started.
  Per-RISC waits and firmware sums are diagnostic, not traced-model latency.
- The completed B32 pair uses identical input hashes and BFP8 precision. Median
  across four ranks: recurrence kernel sum 1161.0 -> 375.6 us (67.6% lower),
  GDN layer 1999.4 -> 1214.5 us (39.3% lower), attention layer 2012.9 -> 2018.3 us
  (0.3% higher). Full traced-model uplift remains pending. Saved raw per-op
  CSVs, all-rank summaries, passed hardware receipts and an explicit comparison.
- All four profiles subsequently completed; physical epilogue testing started.
- Epilogue completed at 22:26 UTC: all eight placement/batch cases passed with
  bit-identical native outputs and clean closure. DRAM B32: 151.93 -> 85.25 us;
  B16: 92.24 -> 64.72 us. B1 loses in both placements and B8 loses in L1, so the
  eventual integration must retain native fallbacks for losing cases.
- B16/B32 combined projections at 32K: 14.8/10.5 TSU, or ideal eight-replica
  aggregate 1900/2692 output TPS. These subtract stage differences from measured
  native full-model TPOT; they are not optimized full-model results. B32 serving
  bucket and aggregate qualification remain outstanding. Saved formulas,
  assumptions, all four profiles and physical epilogue receipts.

## Oct 9, 22:35-22:51 UTC: integrate epilogue and queue targeted Tracy attribution

- Rechecked the eval and perf history after precision questions. BFP8 decoder
  weights reduced short-probe mean relative logit error38.65% ->9.16%; corrected
  GPQA preprocessing was also required for the accepted176/198 result. There
  is no corrected-harness BFP4 rerun proving BFP8 strictly necessary. The
  previous19.15 GPQA mean TSU used optimized BFP4 GDN; native BFP4 measured15.03,
  accepted native BFP8 measured13.01. These variable-length workloads are not
  causal timing controls. Matched native B16 BFP4/HiFi2 gains9.3%/8.6% over
  BFP8/HiFi2 at16K/32K; kernel-policy changes explain a separate larger delta.
- Implemented explicit single_step_shared_qk_epilogue policy/config. B16/B32
  consume raw FP32 recurrence output and use a persistent BF16 output buffer;
  B1/B8 retain the native epilogue and prefill retains chunked scan. Default and
  frozen running model sources remain unchanged. No precision reduction.
- CPU preflight470 tests and40 subtests passed, one unrelated skip. Queued the
  real-weight B32/B16/B8/B1 three-control comparison with64 FP32-reference
  updates and all-rank projected/state bit equality. Device-lock follower
  qwen38-gdn-epilogue-layer-v1-20261009 is persistent, bounded and unqualified.
- Reanalyzed all four existing BFP8 Tracy captures, verifying original CSV
  hashes and every selected rank's row/time totals. AtB32 shared-QK, recurrence
  plus preparation375.63us; largest generic recurrence kernel about159us on
  rank0. Packed convolution149.71us, attention1504.90us. NoC utilization,
  congestion, DRAM utilization, CB waits and per-core min/max fields are empty.
  Reader/TRISC durations include waits, so NoC or placement causality is unknown.
- Added isolated source annotations for reader backpressure, DRAM completion,
  L1 preparation, compute input/math stages and writer output waits. One/two
  input buffers atB16/B32, control/instrumented pairs,24 calls total. Original
  kernels are unchanged; require identical outputs/state and raw timing zones.
  First staging stopped at a CPU regex test that omitted digits in the L1 zone
  name. Fixed it, verified source token preservation, retained the failure.
  v2 passed472 tests/40 subtests and collected the physical test. Persistent
  qwen38-gdn-phase-profile-v2-20261009 waits for the same global hardware lock.
- User asked about30TSU and bandwidth ceiling. Updated the explicit BFP8 traffic
  model: at32K B16/B32,12.884/18.655GB per chip per step. With assumed512GB/s,
  ideal39.74/27.45TSU; current native11.75/7.35 is29.57%/26.79% of that model.
  This is not measured DRAM utilization or a complete roofline. B32/30 needs
  lower traffic or a different TP design; B16/30 remains an unproven stretch.
- Both new jobs use the shared hardware lock,2h/48GiB/8CPU caps,90-minute capture
  deadline,20-minute test limit and process-group cleanup. Profiling additionally
  caps files/output and free disk. Survive disconnect, not reboot. Existing
  perf-priority service remained live on native-before; no new full-model cell
  was completed at the last observation. Source, receipts and bottleneck audit
  are published under the three new evidence directories. Neither queued job
  is a hardware pass, speedup or model-quality qualification.
- Follow-up: fresh32K/B32 completed7.34297TSU and B16 completed11.75371,
  matching the prior baseline. Native-before advanced to16K/B32. After the
  interruption, all four owned service invocations were revalidated live;
  the new physical experiments were still waiting for the shared device lock.


## Oct 9, 23:08-23:24 UTC: completed phase profile and real-weight epilogue

- Native-before finished all four cells and clean shutdown. Refreshed 32K B16/B32
  is11.754/7.343 TSU;16K is12.646/8.040. The persistent full-model queue advanced
  to shared-QK and loaded all64 layers; the native-after bracket and qualification
  remain queued. Do not treat active service Result=success as completion.
- Real-weight epilogue completed all B32/B16/B8/B1 controls. Every projected
  output/state matches bit-for-bit on all four ranks. B32 block1016.36->951.70us
  (+6.79% throughput), B16 710.69->685.47us (+3.68%); native fallbacks unchanged
  within timing noise. Linear48-layer savings3.103/1.211ms are projections.
- Targeted phase profiler completed all24calls, eight buffer/instrumentation
  cases, all ten labels, identical instrumented/control output/state, passing
  JUnit and clean shutdown. Added zones cost about2% in these eager calls.
  Two buffers reduce B32 recurrence234.84->158.95us and B16 132.57->91.67us;
  that setting is already selected by the model candidate, not a new gain.
- AtB32/two buffers, mean per-core unpack input wait9.82us; delta57.42us,
  state update53.74us and output31.41us. Reader L1 preparation65.88us and DRAM
  issue/completion39.92us overlap compute. Writer waits103.20us for state.
  This points toward compute/unpack/pack synchronization and external preparation
  as leads; it does not prove NoC congestion or measure DRAM utilization.
- Parsed2,105,856 relevant raw events, checked exact per-phase work-item counts
  across all four ranks and matched every start/end. No duplicates. Fixed an
  analysis mapping error (global call ID already includes device identity)
  caught by the four-rank count assertion. Archived original rawCSV/Tracy;
  published untouched event rows per call/device under the500KB/file limit.
  Reanalysis reproduces every aggregate and per-core statistic from full CSV.
- Added a physical direct tiled-input preparation test: nine geometry/placement
  cases, all-rank bit equality, two allocations, changed-input trace replay and
  native/fused/native timing. Native exp remains external; model unchanged.
  First staging failed collection because the frozen base omitted prototype
  files; no hardware accessed. v2 explicitly includes them;472CPU tests and
  40subtests pass, one skipped; hardware collection passes. Preserved failure.
- Launched qwen38-gdn-flat-prepare-hardware-v2-20261009, invocation
  c6713def66f8497c9e695154edb2ddbe, under the same global lock. Persistent across
  disconnect, bounded2h/48GiB/8CPUs,90min capture,20min hardware pytest; no reboot
  resume. Source manifest and launch receipts captured. Hardware result pending.
- Network-denied collection/staging helpers retried with sandbox escalation.
  No model promotion, precision change, firmware/NFS mutation or running queue
  rewrite. Original large profiler artifacts remain on host disk and locally.

## Oct 10, 04:16-04:33 UTC: resume Metal, qualify shared Q/K, fix direct preparation

- User dropped the Blaze pivot and resumed BFP8 tt-metal optimization, with20TSU
  as the next goal and32K primary/16K secondary. No Blaze implementation was
  launched. Read current persistent-unit state before acting.
- Completed prior full-model native/shared/native sweeps:32K B16/B32
  14.8707/10.6015TSU, gains26.5%/44.4%;16K16.3371/12.1264. Controls within3%.
  GPQA178/198 (89.90%), five incorrect65,536-output-token cutoffs, all198
  included. Saved-response audit verified all private hashes and usage; completed
  response score also178/198. Copied raw public receipts and manifests locally.
- Container image had all eight TP4 DP workers and passed API/tool smokes.
  OpenBench failed before requests because its staged openbench.py shadowed
  the installed package. Captured this launch failure; no container-eval score.
- Found the physical direct-preparation failure:32-byte scratch spacing violates
  Blackhole DRAM's64-byte matching-address alignment on second-face reads.
  Fixed scratch spacing/indexing without arithmetic or precision changes.
  v3 recovered the dirty device through the safe runner, then all nine physical
  cases passed, including two allocations and changed-input trace replay.
- Added opt-in single_step_flat_prepare_epilogue policy with persistent Q/K,
  value/gate and output scratch. B16/B32 only; B1/B8 paths and prefill unchanged.
  Real-weight native/candidate/native controls: all four ranks identical after
  64FP32 updates. B32 GDN block1016.70->805.58us; B16 710.27->568.64us.
  Projections only:32K16.54/11.88TSU,16K18.38/13.83;20TSU not yet reached.
- CPU integration gates481passed/40subtests/one unrelated skip. Exact-source
  layer rerun reproduced the gain. Launched persistent full-model v1, then
  stopped that owned unit during loading when user requested profiling first.
  Its failed/interrupted status is an intentional reorder, not a model failure.
- Replacement qwen38-gdn-fusion-full-v2-20261010, invocation
  a257a3e921da46a3a9c987347c196190, puts four matched32K B32/B16 Tracy captures
  before full sweeps and full GPQA. At04:32:31 first profile was running. Same
  exclusive device lock;16h/256GiB/16CPU and child/output bounds; survives
  disconnect, not reboot. Preserve every prior source/receipt; no live mutation
  of immutable sources and no promotion of incomplete results.
- Reviewed release PDF and current Shield on-dispatch/release workflows.
  TTIS has the eight-DP experimental Docker/Helm package, but the checked-in
  Qwen models-ci-config entry lists P300X2 only. Galaxy dev-catalog/CI integration,
  successful Shield hardware dispatch/release run, updated candidate image and
  release automation remain. Do not conflate local GPQA with Shield release.
- Modeled useful-byte fractions improved to37.4%/38.6% at32K B16/B32, versus
  native29.6%/26.8%, assuming512GB/s/chip. Not DRAM-counter utilization. Doubling
  means29.74/21.20TSU; graph cleanup, recurrence scheduling and KV delivery need
  end-to-end attribution before promising that target. Stage timelines overlap.
- SSH helper executions blocked by sandbox were retried with explicit escalation.
  Native installation, weights, NFS and firmware were not changed.

## Oct 10, 04:39-04:59 UTC: B16 priority, profile attribution and compact front end

- User priority is now per-user prefill/decode performance at B16, 32K first
  and 16K second. B32 becomes a secondary check; long context stays in scope.
  Added B16_PRIORITIES.md with expected per-op, decode and all-in benefits.
- Collected all four completed current Tracy profiles. At B16, packed
  convolution is 110.46 us/layer, including about 73 us of output tilization.
  Paged-attention inclusive time is 796.77 us/layer. Firmware wait sums are
  not independent costs and blank NoC counters do not establish congestion.
- Started an unselected compact GDN front-end prototype: one convolution row
  per user, native ordered BF16 arithmetic, in-place history updates after
  all old rows are read, and compact outputs consumed by the direct FP32
  preparation reader. Covers aligned 64-byte DRAM reads for odd compact rows.
  No serving policy/default selects this code and no hardware gain is claimed.
- Added a physical TP4 comparison against native convolution/history plus
  preparation, all four ranks, B16 before B32, two allocations, L1/DRAM,
  public/compact projections, and changed-input traced replay. Prepared
  vectors and convolution outputs require bit equality; history must match
  independent host chronology. Native/fused/native timings retain drift.
- First staging stopped before hardware: eight new negative tests omitted
  the repository expect_error fixture's required message argument. Preserved
  v1 evidence. Corrected v2 passed 491 CPU tests and 40 subtests, one unrelated
  skip; hardware test collection passed. Kernel execution remains pending.
- Started persistent qwen38-b16-priority-v2-20261010.service, PID3565241,
  invocation f5b0da5649614052ba7b8ddf37b466d5. It waits for exact current
  fusion/GPQA invocation a257a3e921da46a3a9c987347c196190 to finish cleanly.
  Then B16-only 32K/64K/32K prefill-budget sweeps at 32K/16K contexts, the
  preparation regression screen, and compact front-end tests acquire the
  common device lock in order. Source hashes, deadlines and receipts retained.
- At 04:58:55 UTC both controllers are live. The current control has completed
  32K/B32 at10.60197TSU and moved to32K/B16. No active run was interrupted.
- Audited serving scope: 8 DP x TP4, max_num_seqs16, configured256K per request,
  shared KV pool. Chunked scheduler prefill and prefix caching remain disabled;
  the older BFP4 continuation diagnostic passed but does not qualify the
  current BFP8 plugin scheduler or device sampler. Internal model chunking
  and skipping intermediate output heads are already enabled.
- Sandbox denied the initial collector/stager/rsync SSH attempts; explicit
  escalated retries succeeded. Only task-owned host disk/memory was changed.
  No precision downgrade, deployment promotion, native install or NFS change.

## Oct 10, 05:04 UTC: reload comparison and expected-gain communication

- User requested clearer optimization plans and asked why reload bandwidth is
  higher. Rechecked Blaze source at0ecfc5099203387554a5ca2912f066c098a45fd0:
  its92% figure is an expert-matmul measurement, explicitly not a whole-model
  reload utilization measurement. Bank-local contiguous reads, placement and
  tagged in-flight transactions explain the favorable streaming design.
- Reconciled our read-only499-508GB/s result with the already completed delivery
  sweep275-280GB/s and current model useful-byte estimate37.4%. These are
  distinct measurement boundaries. Delivery cost is proven for the prototype;
  NoC saturation as the model bottleneck remains unproven.
- Added this comparison to B16_PRIORITIES.md. Expected gains remain separate:
  current fusion about11% decode, broader compact front end another10-20%
  engineering target, prefill budget about10% input throughput. The queued
  compact prototype implements only part of the broader front end.
- Both persistent units remain active; the full-model control is running and
  the B16 follower waits for its exact predecessor. All precommit checks pass.
- Reconstructed operation-family costs from the B16 control's exclusive
  operation rows, scaling representative layers48/16 and outer work once.
  Rank-median estimates: matmul18.69ms, SDPA12.59ms, layout/padding/slicing/
  conversion19.49ms, other13.60ms. Explained these as extrapolation, not a
  completed full-trace P0 reconciliation. Useful-byte stage estimates74%/71%
  of peak explain why whole-step37.4% does not mean every reader is that slow.

## Oct 10, 05:13-05:14 UTC: persistent P0 preempts queued optimizations

- User made this profiling P0 and asked for disconnect persistence. Staged
  immutable p0-priority-source-v1 and passed497CPU tests plus40subtests and
  physical-test collection. New ownership tests reject changed PID/invocation.
- Launched qwen38-p0-priority-v1-20261010, PID3582493, invocation
  377cf1d7b3554bc1b3d559e600c15323. Verified active status, parent3530696
  suspended (only orchestration), current child still running, and registered
  identity-checked ExecStopPost resumption. Existing downstream jobs unchanged.
- Queue: current control sweep closes, then B16/32K unprofiled full64-layer
  decode and profiled replay with sampler/history, then resume exact existing
  controller. No running hardware process was killed for this reorder.
- Addressed prior full-capture marker overflow using8192program capacity and
  larger bounded export budgets. This needs hardware verification. Preserve
  raw capture, exact source and output hashes; do not infer NoC utilization
  from wait-inclusive RISC durations or accept incomplete trace coverage.
- Staging first stopped before any mutation because the disk-free assertion
  required64GiB but the dedicated host volume had32GiB. Put large captures in
  removable host tmpfs (284GiB available); control/source remain on host disk.
- At05:14UTC the current control moved to its final16K/B16 cell. Expected
  P0 result35-55minutes if enlarged capture limits resolve the previous failure.
  This is an estimate, not a completed result or reboot-resume guarantee.
- User set70% effective bandwidth as a planning requirement:35.95ms/27.8TSU
  at32K/B16. Added a stage budget requiring roughly31ms savings, mostly graph
  fusion/compact intermediates. Current6.8ms candidate saving is included.
- Rechecked precision history: matched BFP4/HiFi2 native gains9.3%/8.6% over
  BFP8 at16K/32K. No corrected-harness BFP4 GPQA rerun proves qualification or
  proves BFP8 necessary. Keep qualified BFP8 while P0 investigates larger costs.

## Oct 10, 05:35 UTC: P0 hardware passes; offline export recovery

- Full B16/32K unprofiled diagnostic passed, three restored-state replays at
  67.240/67.253/67.240ms, matching the natural-prompt sweep's67.245ms boundary.
  The profiled hardware diagnostic also passed in623.76s, three replays at
  72.240/72.290/72.259ms. Profiling overhead is about7.5%; not a model regression.
- Export exceeded the8-GiB single-file bound before full trace reconciliation.
  Preserved304MiB Tracy capture,23MiB compact device report and8.3GiB partial
  host-zone export. The parent resumed automatically; no hardware hang occurred.
- Inspected native export: it emits every CPU zone then filters TT_DNN/TT_METAL
  in pandas. Added CPU-only recovery that exports TT_ zones directly, retains
  all messages/signposts and the original compact device timings, and omits
  optional host child-function timing. Original capture remains unchanged.
- Launched persistent45-minute CPU-only recovery service
  qwen38-p0-export-recovery-v1-20261010, PID3619025, invocation
  3fa70019ba604d299e9d225a8887a717. It requires all-rank/all-layer coverage and
  exact profiled/unprofiled output hashes before accepting the analysis.
  Hardware qualification is not full trace reconciliation; result is pending.

## Oct 10, 05:39:31 UTC: full-model timing reconciliation succeeds

- CPU-only export recovery completed without another hardware run. Direct TT_
  zone filtering reduced host timing CSV from over8GiB to11.8MB, preserving
  every model/sampler op plus all metadata/signposts and device timing rows.
- Coverage: all64layers,4TP ranks,3replays,5001device-op records/rank/replay.
  Longest device spans71.948/71.957/71.945ms account for host step times within
  0.403/0.461/0.433%. Exact profiled/unprofiled logits and sampled tokens match.
- Profiled operation-family medians: layouts/padding/slicing/conversion22.58ms,
  matmuls18.89ms, SDPA12.63ms, other14.82ms. Inter-operation uncovered gap
  0.336ms, device sampler0.481ms. Instrumented vs unprofiled overhead7.46%;
  do not apply that correction uniformly to each op or infer DRAM counters.
- Preserved full capture/results on host disk underp0-profile-result-v1,
  collected raw CSVs locally and published lossless compressed chunks with
  original hashes/reconstruction order. Broader P0 gate remains incomplete,
  but this requested full-model timing reconciliation is complete.

## Oct 10, 05:49 UTC: consume convolution output in its compact layout

- User asked whether transformations can happen while output streams. Audited
  producer/reader code: current packed convolution retains one of four rows,
  then untilizes/slices/tilizes each Q/K/V output. The compact prototype writes
  one row per user directly; its direct-preparation reader handles the layout.
  It still uses two programs and is not concurrent producer/consumer pipelining.
- Estimated removable output tilization cost is 3.5 ms/model step, about 5.5%
  throughput over the control if no replacement cost, or 6.1% over projected
  combined fusion. This is part of the broader front-end target, not additive.
- Added CPU-only simulator probe and launched persistent unit
  qwen38-gdn-frontend-sim-v1-20261010, PID3632342, invocation
  eec01e6cb8374e278d0eca85641f38cd. B16 public/compact and B32 compact compare
  exact BF16 native convolution, FP32 prepared outputs, host history chronology,
  nonaliasing allocations and changed-input rebinding. Synthetic inputs and
  slow dispatch; physical timing/trace tests remain queued separately.
- Preflight imports and repository checks passed. The simulator was verified
  live with a 60-minute timeout, 16-GiB memory limit and two-core CPU quota.
  Initial local SSH-helper execution was blocked by the sandbox; the explicit
  escalated retry launched it. Only task-owned host disk/memory was changed.
- Fusion hardware comparison remains live; B16 prefill/compact-hardware follower
  waits on its original exact predecessor. No in-flight hardware source changed.

## Oct 10, 05:57 UTC: pipeline scope and simulator boundary

- User asked to overlap matmul and transforms. Confirmed Metal's circular-buffer
  reader/compute/writer pipeline and existing fused activation/output packing;
  cross-op overlap still requires explicit compatible programs/buffering.
  Fusion removes traffic; it does not guarantee all layout work is hideable.
- Isolated packed GDN projection in the candidate's representative-layer profile:
  68.95 us matmul plus 55.87 us surrounding reshaping/slicing/redistribution,
  extrapolating the latter to 2.68 ms over48layers. This is an opportunity pool,
  not a proven saving, and belongs within the broader front-end estimate.
- Four CPU-only simulator attempts exited with unsupported SETDVALID/implied
  source-format handling. The final explicitly fenced diagnostic completed the
  compact convolution call, then failed in compact FP32 preparation. Phase-only
  labels in earlier attempts did not prove executed-kernel attribution; the
  provisional native-layout explanation was not established. No simulator or
  native runtime edits, no numerical pass, and no hardware speed claim.
- Retained each failed source/log/receipt and external service terminal status.
  Fatal simulator exits bypassed Python finalization, leaving stale running
  receipts; terminal service observations supersede them. Hardware full-model
  comparison and the B16 follower are both still active with their original
  identities. B32/32K candidate completed its first two measured repeats.

## Oct 10, 06:10 UTC: B16/32K fusion measurements and full operator roadmap

- Candidate B16/32K cell completed all three measured repeats: 16.5502 output
  tokens/s/user, 264.803 aggregate tokens/s per TP4, 60.4222 ms/token. Decode
  durations were 7.67362/7.67346/7.67391 s for 127 steady steps; output hashes
  matched across repeats. This is about 11.3% above the qualified 14.871 baseline.
- Prefill is 5,323.47 input tokens/s, TTFT 98.631 s, and all-in aggregate output
  throughput 19.265 tokens/s for 32K input / 128 output. Warm model-harness
  timing excludes loading/compilation and HTTP. One TP4, not measured Galaxy DP8.
- Persistent fusion service remained active in the candidate sweep. The final
  after-control and full GPQA are still pending; no serving promotion occurred.
  Retained source/config hashes, raw three-repeat results and snapshot scope.
- Scoped all 38 profiled operation types, reconciled 260 matmul calls against
  all 12 rank/replay records, and generated a reproducible inventory. Encoded
  padded weight bandwidth averages 73.5% of assumed peak; output projections
  are 57%, MLP down 68%, gate/up 83%. These are modeled bytes, not counters.
- Added OPERATOR-ROADMAP.md: finish current qualification, compact GDN layouts,
  retune weak matmuls, then SDPA delivery, recurrence, residual/norm and CCL.
  Prefill budget comparison stays queued; prefill stage profiling precedes
  broader prefill rewrites. Targets are estimates with overlapping ownership,
  accuracy gates and explicit limits; 70% full-model utilization is not proven.

## Oct 10, 06:23 UTC: complete candidate sweep and queue weak-projection tuning

- Fusion candidate completed all four cells and clean device closure. B16/16K
  is 18.3693 TSU; B16/32K remains 16.5502. The live controller moved to the
  final native-after comparison; GPQA is still pending. No serving promotion.
- Implemented 44 bounded real-weight TP4 output/down projection cases across
  B16/B32, including reader counts, storage placement and larger K blocks.
  Pinned native source confirms 1-3 readers and multi-shard block support.
  Preserved weight values, per-user/all-rank dense checks, input-changing trace
  replay and control drift gates; timing includes layouts and collectives.
- Six local contract tests and 16 host preflight tests passed; physical test
  collected. No accelerator was opened by preflight. Created new isolated
  projection-sweep source/control/output directories on task-owned host disk.
- Persistent qwen38-projection-sweep-v1-20261010.service verified active with
  PID3671135, invocation998fcce35571487c9eac759638d71013, waiting on exact
  B16-priority invocation f5b0da5649614052ba7b8ddf37b466d5. Hardware remains
  behind current fusion/GPQA, prefill-budget and compact-front-end qualification.
  Runtime/source installation and all existing in-flight sources unchanged.
- Local SSH staging initially hit the sandbox; explicit escalated retry
  succeeded. Preserved launch, source hashes, preflight output and helper.
- User emphasized 30 TSU. Recorded the 33.33-ms target and 27.09-ms remaining
  gap, useful-byte bandwidth assumptions, larger compact-decoder scope and
  a separate one-draft MTP sensitivity at unchanged B16. Neither the new
  sweep nor incremental savings establish 30 TSU. MTP stays unselected.

## Oct 10, 06:36-06:58 UTC: native-only 30 TSU and complete compact GDN prototype

- User clarified the goal is 30 native TSU at B16/32K, with no speculative
  decoding. Removed MTP from the active roadmap. The current 60.422 ms step
  must save another 27.089 ms; this compact prototype targets only 4-6 ms of
  that gap, within the larger fusion effort, not additive to its broader target.
- Implemented opt-in compact packed projection, convolution/history, direct
  preparation and gated epilogue through output projection/TP reduction. QKV/z
  and gated output stay in compact L1. Native arithmetic and precision remain.
- Added exact physical geometry and alias validation, aligned odd-user reads,
  disjoint compact row writes and padding ownership. Added real-weight changing
  input/state tests and matched full-model controls. Prefill/small-bucket paths
  are explicitly covered by selection tests. No serving default changed.
- Local descriptor tests passed. Host preflight: 512 passed, 69 subtests passed,
  1 unrelated skip; three hardware test entry points collected. These CPU
  results establish contracts/collection, not kernel numerical correctness.
- Launched compact-gdn-v1 persistently after the existing B16 priority queue:
  PID 3704023, invocation 436e57932b7349868ee2d8042b959830. Verified still waiting
  with hardware_started=false. Stopped only the exact waiting projection-v1
  follower, preserving its source, and relaunched as projection-v2 behind
  compact: PID 3704026, invocation 2a7ff1771ed945398736788e16613aa9.
  Current hardware fusion/GPQA and B16 jobs were not interrupted.
- Refreshed native-after evidence: 32K/B16 returned to 14.8715 TSU, supporting
  the current fusion candidate's 11.3% gain. Final 16K control was still running;
  GPQA remained pending. New compact policy has no physical result yet.
- Reviewed frozen-source differences: inherited convolution reader change is
  formatting only; the bounded-runner optional budget does not affect this call.
  Retain manifest and difference hashes instead of claiming exact checkout identity.
- Sandbox denied initial SSH and clang-format's temporary Git index. Escalated
  retries succeeded; formatting modified files and then host preflight passed.
  A documentation patch had stale context and was reapplied after checking that
  it made no partial edit. No native install, firmware or NFS changes.

[Launch, source manifest, tests and remaining gates](../galaxy-evidence/compact-gdn-launch-v1/README.md).

## 2026-10-10T07:21:47.735565+00:00 - Matched roofline reconciliation

- Revalidated active fusion controller PID 3530696 and its qualification child; native G0 was loading replicas, not terminal. B16/compact/projection followers remained live and waiting. No hardware job was restarted.
- Completed B16/32K control drift is +0.003082%, with identical before/after output hashes. Candidate remains 60.422 ms / 16.550 TSU; its accuracy qualification is not complete. Retained completed sweep, control comparison and exact live invocation receipt in `galaxy-evidence/roofline-reconciliation-v1`.
- The saved October 6 specification's 86-TSU B16 ceiling assumes BFP4 weights and 8K. Current BFP8 weights and 32K increase useful modeled traffic to 12.883 GB/chip/step, a 25.162-ms floor (39.743 TSU). Applying its 85%-bandwidth plus 4.7-ms overhead assumptions gives 34.302 ms / 29.153 TSU. The candidate is 1.761x away from that scenario, or 26.120 ms slower. Neither ceiling nor P2 scenario is a combined compute/memory feasibility guarantee.
- Reproduced arithmetic with a hardware-free report script, retaining current padded projection shapes for the BFP4/BFP8 and 8K/32K scenarios. Profiled family sums are not treated as additive wall time or remaining removable work; existing 6.82-ms fusion gain is not counted twice.
- Issues: web open could not retrieve the artifact. Direct shell fetch initially failed sandbox DNS, then succeeded with approved access, but returned only the HTML shell; the shell's public content endpoint returned HTTP 403. Used the hashed saved PDF extraction and explicitly did not claim a refreshed online version. Initial receipt collector hit sandbox SSH restrictions; retried with approved access and collected successfully.
- Priority remains compact/fused native B16/32K dataflow, supported by projection and SDPA tuning. No speculative or precision change, no serving promotion, and no claim of 30 TSU.

## 2026-10-10T07:42:06.603367+00:00 - Shield reference run 38027117236 audit

- Downloaded partial report and benchmark logs; pinned Metal e6b4fe334de4b008ca2cd800f2b26130417b3aa0 and inference-server a44268316063ebf0d25a92f70d769965e17e5942. Extracted the exact Metal implementation without changing checkout. The linked dispatch-validation job is distinct from hardware job 114140267161; the run was cancelled after 22 reported cells. No accuracy pass inferred.
- Confirmed actual run precision from five narrowly retained log lines: MLP gate/up BFP4, down/GDN input/attention BFP8, MLP LoFi. Reference uses TP8, C15/32K, 59.370-ms client TPOT and 68.877-s TTFT. Ours uses BFP8 TP4, B16/32K, 60.422-ms native decode and 98.631-s TTFT. Different chip count, precision, concurrency and timing boundaries prevent a normalized speedup claim.
- Found reusable long-prefill trace/device-offset dataflow, two-link all-gather/MLP/SwiGLU prefill, scheduler admission and run-coalesced state remapping. Verified device-offset SDPA and fuse_swiglu bindings exist in the allocated native checkout. The selected path does not use these prefill forms; generic experimental AGMM code is not equivalent. Native source/install unchanged.
- Exact inference-server commit reports 190 to 59 ms client TPOT after admitting a whole 15x32K burst rather than splitting it. Our scheduler retains a 262144-token admission limit. A prospective experiment must keep the 32768-token device activation budget separate, check host-buffer allocations and quantify TTFT tradeoffs. Neither this change nor remapping improves fixed-batch native decode in isolation.
- Most documented decode gains were ported from our demo and already covered. Retain BFP8: reference teacher-forced findings also show a Qwen3.8 precision loss; they are not GPQA results. No new 30-TSU evidence found.
- Read-only live check confirmed fusion qualification and all three followers active. G0 replica loading continued through 07:38 UTC. No in-flight source, service, or queue was modified. Retained full report, sanitized runtime configuration, source hashes, precision receipts and prioritized scope in galaxy-evidence/shield-reference-38027117236.
- Investigation issues: overly broad log searches produced truncated output; subsequent reads were narrowed. rg is absent on the remote host, so native API checks used bounded grep output. Two guessed local paths were absent and were corrected from rg --files.

## 2026-10-10 17:58-18:04 UTC - Fusion GPQA complete; compact trace test recovered

- Live process checks found fusion and B16 controllers terminal, compact v1 failed, and projection v2 failed without starting hardware. Collected authoritative receipts instead of assuming paused conversation meant running or restarting an observation timeout. Fusion GPQA completed 177/198 (89.39%), 65m54s, concurrency128; six output-budget cutoffs count incorrect. Verified all 34 G0 model/precision hashes against the GPQA controller manifest.
- Completed B16 activation-budget A/B reduces 32K TTFT 98.66 -> 93.09 s with unchanged output hashes; decode stays ~14.87 TSU on the older decoder. No full-eval promotion or combined-fusion performance claimed.
- Compact v1 passed 26 epilogue cases and stationary real-layer comparisons, but changing-input exact hashes diverged at step1. Identified post-capture allocation of the other session's persistent input/state in the harness. Changed only that test: allocate both sessions and zeros, warm both graphs/reset copies, then capture; retain every exact-state/history/output assertion.
- New isolated source compact-gdn-source-v2 passed 512 CPU tests plus69 subtests (one unrelated skip). Physical epilogue and real-layer tests passed. B16/B32 each passed all-rank exact checks at 1/2/4/8/16/32/64 changing-input updates. This supports the trace-lifetime diagnosis without claiming an instrumented exact alias. Native/kernel source unchanged.
- Corrected B16 block is 567.18 -> 350.97 us (-38.1%); extrapolated 48-layer saving10.38ms implies ~19.98TSU from current60.42ms, not a measured full-model result. Source-locked control/compact/control 32K/16K sweeps started persistently at18:02UTC. Compact v2 PID208588 invocation43f1990817724573a9b130ed61f3ff0e; projection v3 PID208592 invocationb8a146f6117847b2afba2ccbe1bee1f4 waits behind it using unchanged projection source.
- Kept prior failures and launch sources, bounded runtime/memory and hardware lock; no native install, NFS or firmware changes. The only recovered queue dependency still requires clean predecessor completion. No compact serving promotion.
- Issues: initial guessed layer/run.log and prefill comparison paths were absent; actual controller log and prefill-comparison.json located. SSH helper initially denied by sandbox, then succeeded on explicitly approved retry. Source hash namespaces differed between controller and G0; verified matching relative paths plus the effective precision override rather than comparing unrelated dictionaries.

## 2026-10-10 18:10-18:16 UTC - Queue compact qualification and matched whole-model profiling

- Fresh before-control B16/32K measurement completed at 16.5473 TSU / 60.433 ms, agreeing with prior 16.5502. The active compact controller then moved to B16/16K; no candidate full-model result is claimed. Thirty TSU still needs about 27.1 ms less per step. Isolated compact-block savings project to ~20 TSU pending the full comparison.
- Added a conditional follow-up that recomputes raw timing/accounting, validates source/policy hashes, precision, runtime and prompts, requires stable controls and identical outputs, and only runs hardware for >=1% improvement at the primary 32K workload. The follow-up's model/config file sets and hashes exactly match the measured source, including additions/deletions checks.
- Winning candidates run unchanged 8-replica G0 and full 198-question GPQA with 65536 output budget, then matched unprofiled/profiled B16/32K whole-model trace replays. A completed low GPQA score remains an explicit accuracy failure while allowing diagnostic profiling. No serving promotion occurs. Output/input identity and host/device timing reconciliation remain separate checks.
- Initial source snapshot lacked the later full-trace helper and artifact-budget support. CPU preflight failed before any queued unit or hardware change. Preserved that failure; a new snapshot includes the required existing profiler helpers and passed 541 CPU tests, 69 subtests, one skip, plus physical-test collection. Kernel/model/native install unchanged. The profiling test now accepts an explicit supported BFP8 recurrence policy rather than silently testing the older shared-QK path.
- Started persistent qwen38-compact-followup-v2-20261010 (PID224830, invocation1a140fb3cc4e4aea92234ad4d38de9d6) behind the exact compact-v2 invocation. Stopped only the identified projection-v3 waiting controller with hardware_started=false, then relaunched unchanged projection source as v4 (PID224833, invocationdd0ad89df8004f78aa4270a047e35f91) behind the follow-up. Active compact execution/source untouched.
- Follow-up has 36h runtime, 256GiB RAM limit, clean predecessor gates and the existing hardware lock. Profile artifacts use a unique /dev/shm directory with bounded file/total size and minimum free space; staging saw 275GiB free. Jobs survive disconnect, not reboot. Retained source manifests, unit invocations, launch arguments, test receipts and live queue snapshots in galaxy-evidence/compact-followup-v2.
- Issues: one helper SSH attempt was sandbox-denied and retried through the prescribed escalation; one broad unit listing was truncated and subsequent checks named only task units. A local shell glob matched no file; corrected to explicit paths. No active benchmark was restarted.

## 2026-10-10 18:18-18:29 UTC - Batched prefill prototype; compact full-model L1 conflict

- Confirmed pinned native support for batch_idx_tensor paged fills and batched chunked SDPA. Added default-off QWEN_PREFILL_BATCHED_ATTENTION path: two cache fills plus one attention call, preserving local row mapping, BFP8 cache casts and causal offset/chunk selection. Local row tensors are setup allocations shared through one replica CCL context; decode and unaligned-prefix paths retain existing behavior.
- Added CPU causal/page-ownership tests and bounded four-rank physical before/batched/after comparison for B2/B3 tails, B16 2K chunks at 0/16K/32K context and B32 1K chunks at32K. Checks exact expected cache writes including inactive pages, exact baseline output, and selected-row dense references for every user/rank. Scope is only fill/attention; no whole-model or GPQA qualification claimed. Engineering target1.25–1.5x boundary, conditional5–8% TTFT reduction if the stage is25% of prefill; zero isolated decode gain.
- Frozen preflight passed556 tests,69 subtests,one skip and collected the physical test. Queued unitprefill-attention-v1 PID238099 invocation9df6962d09c44beabf1037326bb33f2e behind projectionv4 with96GiB RAM/28h runtime and existing hardware lock.
- Compact v2 then failed at18:26UTC during first full-model prefill, before candidate decode timing. Vocabulary-head DRAM-sharded matmul static buffers ended at850944 while a persistent L1 allocation began at825088. Sweep receipt confirms cleanup_completed=true and unitMainPID0. All follow-up/projection/prefill units failed dependency checks with hardware_started=false. This is distinct from the earlier trace-lifetime test failure; no numerical compact full-model result exists.
- Leading cause: compact scratch allocates four persistent L1 tensors per linear-attention layer, accumulating across48 layers while the isolated block only retains one. Investigating shared transient scratch for the serialized layer stack; per-request convolution/recurrent state must stay private. No reboot/reset was needed or attempted.
- Issues: guessed filenames for older MLP/profile helpers did not exist; subsequent searches used actual source definitions. The first scripted SSH reads were sandbox-denied and retried with escalation. Broad native exception text included long C++ stacks; narrowed to Python call site and the allocator error. Failed follower receipts are preserved; none are described as still queued.

## 2026-10-10 18:30-18:34 UTC - Share compact L1 operands within each replica; recover bounded queue

- Python failure site is model._dram_logits during the first full-model prefill, after all64 layers loaded. The sweep records a clean teardown. Compact scratch retained4 L1 tensors per GDN layer:192 per batch shape across48 layers. Introduced CompactScratch shared through one replica CCL context, while FP32 recurrent/convolution state and all other per-layer workspaces remain private. The serialized CQ0 stack consumes each transient output before another layer reuses it. Cross-mesh pools are rejected; separate pools and batch geometries retain independent storage and stable addresses.
- CPU48-layer regression confirms4 L1 buffers atB16 and8 withB32 also prepared, rather than192/384; layer-private workspaces remain distinct. Frozen source passed557 CPU tests,69 subtests,one skip. Physical epilogue and real-weight checks passed again, including exact all-rank changing-input state/history/projected outputs at1/2/4/8/16/32/64 updates forB16/B32. Full-model resolution/performance is still unproven.
- Revalidated exact failed unit invocations withMainPID0, clean failed-sweep teardown and hardware_started=false on all followers. Created compact-gdn-source/control/output-v3, retaining old evidence. The new frozen model includes the default-off prefill prototype; batched attention remains disabled throughout compact qualification. Precision/native install unchanged and no hardware reset performed.
- Restarted compact-v3 PID244902 invocationddc627b6d8ff4f80bcee38abd1294378; conditional GPQA/profile followup-v3 PID244905 invocation28c688b7780445999483523a9cbc75fc; unchanged projection-v5 PID244909 invocation6f7c63955e144aa79693e4d8260cef7c; unchanged prefill-boundary-v2 PID244919 invocation258b685967ee458f84d0d0c809700fad. All verified live. Compact reached fresh before-control model loading at18:34UTC; followers wait for exact clean predecessor receipts. Persistent bounds and hardware lock remain.
- Follow-up reuses the exact compact-v3 source/manifest, preventing harness-source divergence between timing and eval. Corrected copied launch provenance metadata to the actual manifest digest, retaining launch.before-manifest-correction.json; commands, source and live processes were untouched by this metadata correction.
- Current achieved B16/32K performance remains16.55TSU; the compact~20TSU estimate and30TSU target are not measured. L1 accumulation is the leading cause; no exact allocator-address attribution or post-fix full-model success is claimed yet.

## 2026-10-10 18:37-18:47 UTC - Publish compact pool; queue 4K changing-input validation

- Verified compact-v3 PID244902 live and progressing through all64 model layers, then the32K before-control; repeat0 completed. No candidate full-model timing yet. Published the default-off prefill prototype and shared compact L1 pool at210b30ef573 on anatarajan/qwen38-long-context-throughput-20261007 after precommit and all1936 evidence-index hashes passed.
- Extended the existing exact changing-input block comparison with an explicit4096-step mode, preserving the default64-step integration gate. Added finite readbacks and strict report coverage for B16/B32, allfour ranks, recurrent state, convolution history and projected outputs at every power-of-two checkpoint. This compares the compact block against the qualified flat-preparation/epilogue path; it is not an independent dense reference or GPQA result.
- Frozen snapshot keeps all tt/config hashes identical to compact-v3. CPU preflight569 tests plus69 subtests passed, one unrelated skip; physical test collected. Queued compact-long-horizon-v2 PID258205 invocation474d01e736f645718edb4dc861d2fbc6 after the exact prefill-v2 follower. No active source or prior queued process was edited or stopped. Hardware timeout1800s, expected2-10min after existing work,96GiB/8CPU host bounds; survives disconnect, not reboot.
- Refreshed the operator roadmap to measured16.55TSU/89.39%GPQA, block-projected~20TSU, shared-pool recovery and current follower ordering. Marked the October9 backlog as historical rather than leaving its11.75TSU/untested-epilogue assertions as current guidance. Thirty native TSU remains unachieved.
- Provenance check found one inherited frozen-versus-published model difference: convolution reader line wrapping predates clang-format. Verified the complete diff and whitespace-normalized equality, retaining both hashes, the diff and frozen source. All other model/config files match; active and follow-up sources match exactly. No source was changed to conceal the mismatch.
- Issues: the new staging helper initially hit sandbox SSH denial and succeeded after approved escalation. First CPU preflight caught a missing required message argument in six expect_error calls; corrected the test fixture use and retained the failed v1 snapshot/log. No hardware was launched by the failed attempt. Guessed before/run.log and before.log paths did not exist; used the authoritative controller launch path thereafter.

- At18:47UTC, fresh before-control completed B16/32K:16.583725 TSU,60.300082ms,265.339604 decode tok/s/TP4,95.996245s median TTFT. All three measured repeats completed;16K control started. This is baseline repeatability, not a compact-candidate improvement. Retained the partial sweep and queue snapshot; allfive named units still had live PIDs.

## 2026-10-10T18:54:21.627950+00:00 - Add compact-L1 projection tuning boundary

- Source review found the old output-projection sweep used expanded [B,1,1536] DRAM operands while compact GDN produces [1,1,B,1536] L1. The older declared boundary was valid but would include costs absent from the proposed decoder. Added separate compact-L1 output, compact-L1 down and public-DRAM output brackets, each at B16/B32 with10 configurations plus repeated control:66 measured cases/54 comparisons. Upload checks exact logical/padded geometry and memory; validator rejects mixing layouts or missing coverage. Full-model1.5-3.5ms tuning target remains an estimate, not new savings.
- Retained quantized-weight identity, local/reduced dense reference, changed-input A/B/A and control-drift gates. New boundary includes layout+projection+CCL; setup and readbacks remain outside timing. Expected15-45min accelerator time after predecessors, existing6000s bound.
- Frozen preflight572 CPU tests plus73 subtests passed, one unrelated skip; physical test collected. Model/config hashes match compact-v3. Audited exact waiting projection-v5/prefill-v2/long-horizon-v2 invocations with hardware_started=false, stopped only those controllers, and archived before/after receipts. Active compact-v3 and qualification/profile-v3 were untouched.
- Launched projection-v6 PID268165 invocation960d702be457402a92d32d88b4a4a591, unchanged-prefill-source v3 PID268168 invocationd7501d6753824d31846a13ecef289c32, and unchanged-long-horizon-source v3 PID268172 invocation6f0ee1599ae144d6a6843b49bf22455b. Verified allfive current jobs live. Their original bounds, hardware lock and exact predecessor gates remain.
- Compact-v3 completed its before-control cleanly:32K/B16 16.583725TSU and16K/B16 18.405926TSU. Candidate full-model arm started18:50:25UTC and was loading layer30 at18:52:57. No candidate speed or L1-resolution claim yet.
- Issues: initial staging SSH attempt was sandbox-denied; prescribed escalated retry succeeded. Black formatted three harness files before the frozen preflight. No model/native/kernel edits or serving promotion.

## 2026-10-10 19:03-19:10 UTC - Compact full-model improvement measured; GPQA queued

- Compact completed both B16 contexts with clean teardown at19:06:43UTC. Three measured repeats give32K 20.034955TSU/49.912766ms, versus16.583725TSU/60.300082ms before (+20.81%);16K22.750585TSU/43.954915ms, versus18.405926TSU/54.330329ms (+23.60%). Every before/candidate warmup and measured output hash matches at each context. Recomputed medians and verified raw model/policy hashes against the frozen manifest. No trace capture occurs in measured samples.
- The shared L1 pool now passes the formerly failing full-model prefill/head boundary and both candidate cells. After-control remains active, PID244902, and reached layer33 loading at19:09UTC. This supplies post-fix full-model evidence but does not qualify compact accuracy or prove an instrumented allocator-address cause.
- Verified persistent GPQA follower PID244905/invocation28c688b7780445999483523a9cbc75fc, exact source/manifest and dependency. It recomputes all three raw arms before eight-replica G0/full198GPQA at65536 output budget. The prior unchanged pipeline used46.5min for G0 and11.3min endpoint startup plus65.9min GPQA; estimated compact final score in2-2.5h from19:10UTC, conditional on clean controls/startup. No duplicate job, active-source edit or serving promotion.
- Retained first-repeat, intermediate and completed-candidate snapshots in galaxy-evidence/compact-first-decode-v1. Projected eight-TP4 output is2564.47tok/s at128active users, not measured Galaxy/HTTP throughput. At the user's $2/M output,$10/Galaxy-hour,20% achieved-output-utilization assumptions, output revenue is$3.69/h versus$10/h hardware; output-only break-even is54.16% utilization. Input revenue, serving/prefill duty and other costs are separate.
- Scoped prefix caching/SSD against actual plugin b7e4292e4193cba20abe9c7c68ce489201b2e36b/vLLM0.26.0, not the older local inference-server submodule. Required hybrid recurrent-state cache groups and aligned snapshot/restore are missing; Qwen explicitly disables APC. Proposed existing-framework reuse plus a TT transfer adapter, with146.8MiB recurrent/conv checkpoint per TP4 frontier. No cache implementation or performance claim was added.
- Issues: initial read-only collector SSH was sandbox-denied; escalated retry succeeded. Guessed LOGBOOK and plugin paths were corrected using tracked filenames. Broad raw JSON reads were truncated; published metrics were separately recomputed from complete retained files. No firmware/native-install/NFS changes.

## 2026-10-10 19:23-20:46 UTC - Compact controls pass; queue register-resident recurrence

- Compact before/candidate/after finished cleanly at19:23UTC. B16/32K gives16.583725/20.034955/16.584061TSU;16K gives18.405926/22.750585/18.406055. All output hashes agree; control drift0.00202%/0.000702%. Current32K step49.912766ms still needs16.579433ms removed to reach30TSU. No spec decode or lower precision was introduced.
- Eight-replica G0 passed20:09:50UTC. API checks passed; full198GPQA began20:19:34UTC, sampled temperature1/top_p0.95/top_k20/seed42 with65536 output budget. The20:46UTC receipt has167 completed,159 correct,zero truncations. This is partial progress, not a final qualification score; long answers remain. The source-frozen GPQA/profile queue was not interrupted or edited.
- Implemented default-off resident_state GDN compute: retain four FP32 state tiles through the ordered decay/update/output calculation in full-DEST mode. Reader/writer ownership and state precision are unchanged. Removes repeated state unpacks, duplicate decay multiplies and delta CB roundtrip. Engineering target0.5-1.5ms/full step (~1-3%); full-sync and delayed writeback may erase the gain. No model policy enables it.
- Simulator control-first v1 and candidate-first v2 both compiled their first kernel and exited1 at unsupported SETDVALID/src-format interaction before any arithmetic comparison. Preserved raw logs and terminal unit states; stale probe state=running is not a live execution. SFPLOADMACRO-disabled simulator compilation is not native hardware proof. Compiler also emitted existing noc_async_read_tile deprecation warnings.
- Added physical four-rank B16/B32 changing-input and B1/4096-step independent dense-reference checks, exact control/candidate state/output, rebinding, input immutability and matched B16/B32 timing with state reset after capture. Validator checks all ranks, thresholds,508-update timing trajectories, hash agreement, drift and cleanup. Frozen preflight579 passed,1 skipped,77 subtests; physical test collected and shell syntax passed.
- Queued qwen38-gdn-resident-hardware-v1-20261010 PID334725 invocation9a87b6620b6c4892b407a17f7c54f04e after exact compact-long-horizon-v3 invocation6f0ee1599ae144d6a6843b49bf22455b. Existing order remains GPQA/profile -> projection-v6 -> prefill-v3 -> long-horizon-v3 -> resident recurrence. Physical test bounded1800s,16GiB/8CPU,shared lock; waiting unit28h,disconnect-persistent,not reboot-persistent. Confirmed waiting hardware_started=false at20:46UTC. No automatic model promotion.
- Separately published the hybrid prefix-checkpoint/storage foundation in anatarajan/qwen38-prefix-offload-20261010 atfab35e060a581789c95072fd6120e8edfbc6dcb6:21 CPU tests passed. This is codec/storage groundwork, not TT device transfer, vLLM APC integration or an enabled serving feature. Decode/GPQA retain hardware priority.
- Publication audit found31 historical log/CSV entries whose local bytes matched the old evidence index but were excluded from Git by ignore rules. Explicitly restored those exact verified paths to this branch's publication, as already done for the prefix branch. No runtime behavior changed. Raw new receipts are under galaxy-evidence/gdn-resident-v1; new logs/XML are gzip-compressed.
- Issues: read-only collector SSH hit sandbox denial and succeeded with prescribed escalation. Initial guessed ROADMAP.md did not exist; used tracked OPERATOR-ROADMAP.md. A broad queue/file listing was noisy; narrowed later collection to fixed non-private evidence paths. No dataset/raw response publication, native-install/NFS/firmware modification, reset or reboot.
