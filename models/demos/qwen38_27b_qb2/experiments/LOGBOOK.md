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

## Remaining gates and next experiments

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
