# Exact-tree hidden RMSNorm fusion audit

2026-11-13. CPU/source inspection and isolated candidate/probe implementation only. No device, TTNN/torch import, native compilation or timing by this audit. The only edited artifacts are this report and the two new opt-in candidate/probe files named below.

## Recommendation and current evidence

**A two-program exact-tree family is structurally feasible:** combine the existing fold and inverse programs into one32-worker operation, directly scatter folded rows into private inverse-worker CBs, then run the existing80-worker output program unchanged. This removes the globally addressable128KiB folded tensor and one launch per norm without changing statistics, input/output layouts, or mesh collectives. It adds a local admission barrier and completion credits; these may erase its benefit.

A one-program extension can retain the same32 fold workers, B inverse workers and80 output workers, passing inverse rows through local NoC. It removes the remaining4KiB inverse tensor and second launch. Its additional B×80 fanout/credits make it a second experiment, not an assumed improvement. Neither proposal restores a replicated inter-layer residual: both begin after the existing gather-first operation and return the same local normalized5120-wide projection input.

The latest applicable artifact is `perf/integrated14_profile_l0_b32_s33/decode_perf_report.txt`, with `metrics.json` source pin `multichip_decoder.py=4c56cbadb7768054100f921ced79d725a57bfa477beca9648f8f2d932e9bd765`, frozen optimized decoder pinned below. It reports both hidden norms as approximately3us fold +5us inverse +6us output, plus roughly1us gaps per operation. Whole profile is573.6025us device kernels +24.441us gaps =598.0435us. These are merged instrumented observations, not an uninstrumented prediction. The profile does include the accepted local affine helper; its older superseded reshape chain is absent.

Full correct frozen SHA-256: `315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`. All inherited kernel references below use this immutable source. The compact beta/decay and readout norm/gate probes now passing elsewhere do not alter this hidden-norm path.

## Existing contract and exact arithmetic

Source: `tt/multichip_decoder.py:_norm` around499, selecting `super()._norm(self._collective(x),key)` for decode; `tt/optimized_decoder.py:_norm` around611 selects `_norm_zero_meanhidden_rmsnorm_sfpu` for BF16 `[1,B,5120]`, `1<=B<=32`, FP32 gamma `[1,1,5120]`. The wrapper at6363 invokes three generic programs. This is the GDN exact-tree path; full-attention's sharded native RMS is a different policy and must not be silently replaced using this proof.

Let `d=32*r+c`, with `r,c` in0..31. Fold core r loads input tile pages `r,32+r,64+r,96+r,128+r`. For each user u it computes:

`F[u,d] = ((((x[u,d]^2 + x[u,1024+d]^2) + x[u,2048+d]^2) + x[u,3072+d]^2) + x[u,4096+d]^2)`.

Every square and add is a separate FP32 SFPU operation in that order. The inverse worker reconstructs a dense32×32 FP32 tile with `dense[r,c]=F[u,32*r+c]`, executes **REDUCE_COL then REDUCE_ROW**, multiplies by the exact supplied FP32 reciprocal of5120, adds full FP32 epsilon, then uses the existing rsqrt. Keep these source spans and direct-FP32 unpack settings unchanged. Do not use the head128 native FPU norm helper or its truncated epsilon here; that is a different contract.

Output computes the existing negative-zero centering sequence, `(x + (-0))*inverse`, then FP32 gamma row multiplication, then one explicit Float32→BF16 typecast before pack. `_hidden_normalize_zero_` around6318 deliberately preserves the negative-zero instruction path. No intermediate BF16 norm boundary exists in this hidden norm; adding one would be a semantic change. Final nonlogical rows are explicitly zeroed by the output writer.

Source anchors:

- Fold reader/compute: `_norm_zero_meanFOLD_READER`6109, `_norm_zero_meanFOLD_COMPUTE`6143.
- Dense gather/inverse: `_norm_zero_meanINVERSE_READER`6168, `_norm_zero_meanINVERSE_COMPUTE`6225.
- Output reader/writer/compute:6253/6281/6301.
- Wrapper and CB/worker construction:6363–6505.

## Baseline traffic and buffer ledger

All counts below are **per chip per norm** and count API data requests, not physical NoC packets. B is logical batch. Input/output are DRAM; the folded and inverse tensors are already interleaved **L1** by default.

| Phase | Worker count | Requested data movement | CB allocation/core |
|---|---:|---|---:|
| Fold |32|160 DRAM reads×2048B=327,680B;32 L1 writes×4096B=131,072B|5BF16 input tiles +1FP32 folded tile =14,336B|
| Inverse |B|64 L1 reads/user×64B =4096B/user;2 row writes/user×64B; tail padding fills inverse tile to4096B|dense/inverse/zero, three FP32 tiles =12,288B|
| Output |80|160 DRAM x reads×2048B;160 DRAM gamma reads×4096B;160 inverse L1 reads×1024B;160 output DRAM writes×2048B|2BF16 x +2FP32 gamma +1FP32 inverse +2BF16 output =20,480B|

Totals: DRAM traffic is1,638,400B independent of B because full tiles/padded rows are processed. L1 scratch traffic is `131072 + 4096*B + 4096 + 163840` bytes, or430,080B atB32 and303,104B atB1. The baseline's gamma reads include padded gamma rows, but this audit does not change those reads. Baseline read/write barriers: fold one each/core; inverse one read barrier/user and one write barrier/user; output one each/core. Program sequencing supplies two inter-program producer/consumer fences.

Do not count this family as removing DRAM scratch traffic. L1 scratch was already the kept norm179 improvement, and output80 workers already replaced the old32-worker/five-tile output assignment.

## Two-program family: fold + inverse, unchanged output

Launch a fixed32-worker role map; a rectangular8×4 logical grid is a convenient candidate, while retaining the original first32-worker map is a useful placement control. Use host-provided physical/virtual coordinates appropriate to the active mesh device. Do not assume logical coordinates equal NoC coordinates.

Each worker r retains the exact five-full-tile fold. Workers `u<B` additionally own one inverse result. Allocate a common CB descriptor layout on all32 workers, so remote dense-CB addresses have an explicit identical local-offset contract; do not infer them from an unrelated tensor allocation:

| CB role | Capacity |
|---|---:|
| raw fold input |5BF16 tiles =10,240B|
| folded output |1FP32 tile =4096B|
| inverse dense receive tile |1FP32 tile =4096B|
| inverse result |1FP32 tile =4096B|
| inverse padding-zero scratch |1FP32 tile =4096B|
| Total uniform union |26,624B/core, 851,968B over32 cores|

The zero scratch can later be removed or reused, but retain it for the initial arithmetic/ownership control. No live DST state crosses a release or a phase boundary; folded and inverse values are packed FP32 before handoff exactly as before.

Protocol:

1. Reader reserves input CB and, for inverse roles, dense receive CB before advertising READY. Use program-local semaphores initialized by the program launch. A designated coordinator waits for32 ready credits then releases the32 fold writers. This explicit admission prevents a sender writing remote scratch before the receiver establishes this invocation's ownership. Never clear a semaphore after peers can begin incrementing it.
2. Reader loads the original five input pages and publishes them. Compute performs the literal fold and publishes one folded tile. Writer waits for that tile and admission.
3. For every user u and face f=0,1, fold writer r sends64B from folded tile row u into inverse core u's dense tile row r. Source offset is `((u//16)*512+f*256+(u%16)*16)*4`; destination offset is `((r//16)*512+f*256+(r%16)*16)*4`. All requests have64B-aligned endpoints. The32 senders cover every dense byte exactly once; there is no overlapping write or floating reduction on NoC.
4. Each sender completes all data writes with its NoC write barrier, then increments DONE once on each user receiver. Finish the atomic stream before kernel exit. A receiver waits for32 DONE credits, then its reader publishes the dense CB to compute. This is the only point at which inverse arithmetic can start.
5. Inverse role executes the unchanged dense reduction/scalar source, packs its result, and uses the original inverse writer to publish its disjoint row into the existing4KiB L1 inverse tensor. The last logical user owns inverse padding exactly as before. Normal program completion precedes the unchanged80-worker output program.

Avoid a deadlock in role-sharing: fold writer must scatter and signal before waiting for its own inverse result; compute must publish folded output before waiting for dense input. The reader must not wait for DONE before publishing raw fold input. These three orderings are a testable abstract dependency graph.

NoC/accounting relative to baseline:

- Remove131,072B folded-tensor writes and `4096*B` folded-tensor reads.
- Replace them with `4096*B` direct local writes:64×B transfers, or2048 transfers/131,072B atB32;64 transfers/4096B atB1. **Net data payload saved is131,072B per norm at either batch**, before protocol packets.
- Admission uses32 READY and32 RELEASE credits for a simple unicast control. Completion uses32×B credits. AtB32 that is1088 total credit operations; atB1 it is96. Multicast release is optional only after a coordinate/receiver proof.
- Each fold writer needs one data-write barrier before sending completion credits and one atomic barrier before exit. Receivers' DONE waits replace64 individual reads plus their read barrier. The final inverse writes/zero padding remain unchanged.
- Remove one generic-op launch and128KiB global folded allocation; add local CB union and local semaphores. Three semaphore IDs (READY coordinator, RELEASE, DONE) suffice for the straightforward one-shot protocol. Allocation alignment, rather than the4-byte value alone, sets their L1 footprint; measure actual program allocation metadata.

This is not the previously rejected two-pass20-partial arithmetic: the tree and dense mapping stay exact, and no sums are reassociated. It is a direct movement/scheduling experiment.

## One-program extension: preserve output parallelism

Do not fuse by letting each original user worker normalize all5120 channels alone. The prior per-user implementation already incurred320 small x reads and320 gamma reads per user and motivated the kept distributed tree. A one-core-per-user restoration needs a new movement proof to justify repeating that family.

Instead launch80 workers. First32 fold; firstB also inverse; all80 own two output tiles, as in the accepted output phase. Add distinct CBs for output x, gamma, received inverse rows, and output. A conservative uniform descriptor union is:

- fold input10KiB, folded4KiB, dense4KiB, scalar inverse4KiB;
- output inverse4KiB, x4KiB, gamma8KiB, output4KiB;
- total42KiB/core (43,008B), 3,440,640B across80 workers, plus semaphore/runtime/program storage. Optional separate zero scratch adds4KiB/core. This is a planned budget, not a measured compiled allocation.

After each inverse user computes, send its logical inverse **left-face row** to the corresponding row of every output worker's inverse CB. Only column0 in left faces0/2 is consumed by `_hidden_normalize_zero_`; the right-face row currently written by the standalone inverse writer is dead for this consumer. A64B row transfer preserves the complete left-face row and avoids inventing a subword alignment contract. Every output core waits for B completion credits, then publishes its inverse CB and runs the original two-tile output compute/writer. No final global inverse tensor is needed.

This adds B×80 data writes and B×80 completion credits: atB32,2560×64B=163,840B, exactly the baseline output's inverse-read payload, but far more tiny transfers/credits; atB1,80×64B=5120B. The old4096B inverse-tensor publication is also removed. Admission must cover all80 output-CB owners before any inverse fanout;80 ready/release credits suffice for a conservative unicast startup. Fold-scatter completion remains32×B. No kernel may reset future-generation credits, and all outgoing atomics must drain before return.

One-shot per-core CB descriptors and semaphores are recreated/reset by program launch. Confirm that generic-op program-cache replay preserves this initialization, then stress consecutive invocations; do not borrow persistent CCL scratch protocols blindly. Output x/gamma can be prefetched on the reader NoC while compute waits, but add that overlap only after the simple dependency graph passes. Keep reader and writer NoC/processor ownership explicit so independent issue paths do not race shared command state.

Invalid user rows may have uninitialized inverse entries if the final writer zeroes all invalid output rows. Do not assume this safe without poisoned-padding tests and a source proof that output normalization does not mix rows. The conservative initial version can initialize all inverse receive CBs; account its extra local-zero traffic separately rather than hide it in the saved-payload figure.

## Prior rejected/kept evidence and scope

`doc/optimized_decoder/AUTOFIX_hidden_rmsnorm_exact_tree.md` records the prior reassociated20-partial norm passing isolated precision but failing512-step PCC: .9937332 versus .9958222 for the accurate incumbent. That is direct reason to keep the five-square fold, physical dense mapping, COL→ROW reduction order and all scalar boundaries. It is not a rejection of launch fusion with that tree unchanged.

`AUTOFIX_hidden_norm_l1_80.md` and `AUTOFIX_hidden_norm_zero_mean.md` identify already-kept changes: L1 scratch,80 output workers, and exact negative-zero specialization. These must remain in the baseline. `ag_adapted9_summary.csv`/work_log205 already compare distributed-norm/separate and fused AGMM coherent finish families and reject them on end-to-end latency. This proposal does not revisit collective placement or AGMM. It operates entirely inside the selected gather-first hidden norm.

The mesh remains TP4, residual boundary BF16 hidden-sharded1280, no inter-layer collective added. Gamma remains FP32. Context/KV layouts are unaffected. If helper-owned persistent buffers are introduced, owner accounting must be updated; program-local CBs are not persistent model state. Initial scope is compact decode only, with existing prefill/public nonaligned input handling preserved.

## Expected opportunity and discriminating proof

AtB32, current fold+inverse is about8us per norm plus its gap. The two-program candidate can at most remove redundant movement/launch work within that region; it still must compute fold+inverse. Eliminating that entire region unrealistically would save only about16us across both norms, approximately2.7% of598us instrumented decoder time, before gaps. One-program fusion targets all~28us of norm kernels plus their gaps, but preserves all math and output traffic. These are opportunity bounds, **not expected latency gains**. This family cannot alone close the roughly190us reduction needed to turn .594ms into .280/.70≈.400ms.

First prove the two-program family independently. Retain original fold/dense/inverse diagnostic readbacks outside timing; compare reconstructed dense tiles and inverse logical entries bitwise, then final BF16 output bitwise. Use actual input and post-attention gamma, B1..32, signed zero, near-tie BF16 mantissas, sparse impulses across each of five1024-channel chunks, large/small magnitudes, and poisoned inactive input/inverse rows. The existing exact-norm comparator's54 cases are a good starting point, not a substitute for TP4 rank/user/address coverage.

CPU checks should exhaustively cover every scatter byte, disjoint ownership, complete1024-element dense reconstruction, role dependency order and credit counts. Check the original and8×4 worker maps. Device proof needs watcher, B1/B3/B17/B32 changed-input trace, retained old buffers/new addresses, queued invocations with deliberate sender/receiver skew, and failure capture before any reset. Run hundreds of invocations to expose semaphore reuse, and require exact eager/trace parity before whole-layer512-step gates. Negative controls should reject unknown grids, bad output aliases and incompatible dtypes/layouts.

Measure matched native3-op, fused2-op and later fused1-op complete norms with stable outputs and every program warmed before capture. Instrumentation and timing stay separate. A rejected1-op fanout does not reject the cheaper2-op protocol; a slow one-user ownership control does not reject the retained80-output-worker family. Only after exactness and total norm timing improve should the candidate enter whole GDN decode and the stage's unchanged accuracy/stress/runtime/ownership gates.


## Implemented bounded candidate and reusable-epoch source proof

`tt/hidden_norm_fused_candidate.py:hidden_norm_fused` now implements only the two-program family. `tests/probe_hidden_norm_fused.py` provides the independent frozen three-program oracle. There is no one-program fanout implementation and no production integration. The implementation retains the first32 native logical worker coordinates (11-column row ordering on the target grid), maps them through `worker_core_from_logical_core`, and retains the exact native80-worker output descriptors. All32 roles reserve the dense receive slot, including inactive inverse roles, making the uniform-address contract explicit.

Source proof for repeated enqueue and traced replay:

- `tt_metal/api/tt-metalium/program_descriptors.hpp:89` defines program-local semaphore IDs/core ranges/initial values. Three descriptors cover the same32 cores with initial value0. `ProgramImpl` adds these exact descriptors when constructing a program; no persistent global semaphore allocation is used.
- `tt_metal/impl/program/dispatch.cpp:1264` (`SemphoreCommandGenerator`) emits each semaphore's initial value at its configured offset into the batched program configuration writes. `assemble_device_commands` at2442 places these writes into `program_config_buffer_command_sequence`, together with CB configuration.
- The command enumerator at3095, specifically3122, emits the program configuration buffer unconditionally before launch/go messages. The `send_binary` cache branch applies only to binaries; it does not omit semaphore initialization. Thus a cached program invocation starts another initialized epoch rather than inheriting previous DONE/GO counters.
- `tt_metal/distributed/fd_mesh_command_queue.cpp:1844` writes that same command sequence into the trace recorder. The captured stream contains the initial-value writes, so every replay resets semaphores too. Trace command update handles config-buffer ring slots and completion counts. Program reset is performed by dispatch configuration, never by a worker after its peers can start incrementing.
- `tt_metal/impl/program/program.cpp:2417` allocates each local circular buffer once over its entire `core_ranges`, assigning a single address for that descriptor. All five CB descriptors cover exactly the same32 workers; the dense CB has the same local address on each. No separately allocated tensor address is guessed. Every kernel on each role uses the same local CB configuration.
- Kernel READY is sent only after dense reservation, raw input read barrier and raw CB publication. The coordinator waits for32 READY credits before atomically releasing all32 writers. Every READY/GO atomic stream is drained. Writers wait for GO and folded output, scatter complete64B rows, issue a data-write barrier, then increment each receiver's DONE exactly once and drain the atomic stream. A receiver publishes dense input only after32 DONE credits. All outgoing atomics complete before kernel exit; no preceding epoch can leave late credits for a subsequent dispatch reset.
- Program completion across these participating cores separates invocations on the serialized CQ. The selected contract does not claim concurrent use of one caller-owned inverse/output buffer on different CQs; the probe uses CQ0. The final output program follows normal queue dependency handling.

The global config-write/launch ordering is the existing tt-metal dispatch contract relied on by every program-local semaphore; this candidate adds explicit local admission rather than implementing an independent cross-launch reset protocol. There is no unresolved source-level reset race under that contract. Hardware watcher/queued/replay tests remain mandatory: source reasoning does not demonstrate runtime behavior or compile validity.

The arithmetic source transformer imports frozen string constants at device-function invocation, with no runtime AST or source-file reads. The five-square fold and inverse scalar/reduction bodies are reused literally. Inverse changes only CB0→CB4 and CB2→CB5 plus explicit BF16→FP32 source reconfiguration between phases. The inverse writer body changes only data/zero CB indices and reuses the native disjoint row/padding ownership. All three final output kernel strings remain byte-identical. This preserves the intended arithmetic, but only measured comparison can establish exactness.

CPU validation command (completed):

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_hidden_norm_fused --cpu-check
```

Result: PASS,32 logical batch descriptor cases,26624 CB bytes per fused worker,20480 per output worker. Checks cover literal arithmetic spans, output source identity, FP32 unpack modes, physical-coordinate runtime arguments, unique complete4096B dense receive ownership for every user,64B alignment, complete disjoint inverse row/padding ownership,160 output pages, semaphore counts/initial values, producer-before-wait order, and invalid batch rejection. No `torch` or `ttnn` module was imported. Black formatting and CPU checks were rerun after formatting. Source/protocol SHA-256 provenance is emitted in probe JSON.

Parent launch commands below assume its established device reservation/health environment and working directory. Do not run concurrently with another device owner; use a fresh output path each time.

```bash
TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_hidden_norm_fused --batches 1 3 --modes normal --require-exact --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/hidden_norm_fused_smoke_v1.json
TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_hidden_norm_fused --all-batches --modes normal zero scale --require-exact --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/hidden_norm_fused_wide_v1.json
env -u TT_METAL_WATCHER -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE -u TT_METAL_DEVICE_PROFILER python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_hidden_norm_fused --batches 1 3 17 32 --modes normal --samples 7 --require-exact --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/hidden_norm_fused_perf_v1.json
```

Probe coverage: rank/user-distinct BF16 input, true FP32 gamma (synthetic, not model-loaded), normal/signed-zero/large-small scale cases, all32 batches when requested, bitwise final-output diagnostics and logical inverse-col0 diagnostics, unchangedPCC>=.995 floor, rebound input/output addresses,50 queued native/candidate generations with unique retained outputs,50 replay groups per changed input plus50 individually alternating-input replays, and stable-buffer matched full-chain timing. All retained device allocations, native/candidate warmups and copy warmups precede first capture. Both trace handles survive until all comparisons/timing finish; no retained device allocations are introduced between capture/replay. Native has no added output copy or substitute norm. Timing is skipped under watcher/profiler/allocation tracking. JSON explicitly records tracking/cache-exemption environment and source hashes.

Limits before hardware acceptance: no compiler or watcher run here; no exactness or performance claim; final output comparisons cannot directly expose every nonlogical padding bit, and synthetic gamma is not a substitute for model PCC/stress. The transfer model predicts128KiB less payload but1088 atomic credits atB32, so this straightforward family may be slower. If so, profile its ready/scatter/credit phases before adapting (for example, aggregated completion or coordinate-aware multicast release) rather than declaring all exact-tree fusion infeasible. Whole-layer quality gates and context accounting remain the parent's responsibility before integration.

## Adjacent recurrence-epilogue family handoff

Parent reports the existing head-strided standalone norm/gate candidate passed97 exact cases, but uninstrumented matched trace medians were B1 native14.055us/head34.052us/tile32.796us and B32 native24.959us/head41.179us/tile58.051us. These are parent-provided results, not measurements performed by this audit. The cause under investigation is the current readout's head-strided64B reads (256 reads/head atB32). A coherent next family is to have recurrence's writer emit FP32 head-major `[12,B,128]`, preserving state/readout arithmetic, then let the norm/gate owner read four full tiles per head. Own inactive-row padding and retain the final compact `[1,B,1536]` consumer layout; do not restore the old head-strided layout immediately before the fused consumer. No norm/gate or recurrence source was changed by this hidden-norm task.
