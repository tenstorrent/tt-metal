# Multichip decoder work log

Stage4, Qwen/Qwen3.6-27B, pinned HF revision8f33b98937c6031bb49e713dee3e8e185aafd26f.
Starting branch/commit recorded by git; clean initial worktree at0915d931.
Scope: multichip_decoder.py, model tests and docs only. Never push.

2026-11-13: startup read installed multichip, device-usage, model-bringup;
plugin inventory/config confirms tt-autodebug0.1.8 enabled; model-bringup0.1.23.
Environment uses python_env and /home/ttuser/qwen36-27b/lane-env.sh.
Read tech_reports/LLMs/llms.md section3.3, optimized source and its final README.
fuser /dev/tenstorrent/* found no owners. `timeout60 tt-smi -ls --local`
returned four P300c chips; device_inventory.log. Bounded90s FABRIC_1D 1x4
open/close succeeded; topology_probe.log. No reset needed.
Mesh plan written before implementation. No capability reduction authorized.

Initial implementation/tests:
- Added MultichipDecoder subclass of the frozen optimized baseline. Local QKV,
  GDN, MLP, recurrent and KV ownership follows mesh_plan.md; inherited device
  matmul/norm/rope/prefill handling remains the baseline.
- Added run_multichip_decoder: separate-process optimized reference and TP4
  comparisons on identical real weights, input, shuffled page table and positions.
- Baseline layer0 B1/S33 passed (baseline_l0.log/json; reference ref_l0.pt).
- tp4_initial_l0.log: test harness passed None for physical_device_ids, rejected
  by Python binding before mesh open. Fixed to the default empty list.
- tp4_initial2_l0.log: mesh opened; setup called convolution exchange-core
  helper with MeshDevice instead of CoreCoord. Corrected to compute grid.
  Devices closed normally. No model correctness result obtained.
- Requested operator reset authorization via asynchronous question before the
  next device test: multichip skill requires resets after failed multichip runs,
  device-usage skill requires operator authorization. Device work is waiting;
  no reset or further hardware use has occurred while this question is pending.
- Host compileall passes. audit_multichip_runtime.py resolves subclass/base/super
  methods:26 reachable functions, zero direct runtime host conversions/imports.
  Runtime sentinels and native/CCL checks remain pending.
- Capacity arithmetic:15,844,720,640 bytes/chip including4GiB reserve at B1,
  context262144; capacity_plan.json. This is a plan, not a runtime allocation
  measurement. context_contract.json records validation pending and no reduction.
- Pre-commit hooks run over new code, hooks.log. Re-run after subsequent edits.
- AutoDebug launcher sandbox preflight failed because unprivileged namespaces
  are disabled. Retried with AUTODEBUG_SKIP_CHILD_SANDBOX=1 under the existing
  explicit subagent authorization/unrestricted environment. Child is read-only,
  no devices; autodebug_launch_retry.log.gz. Diagnosis pending.

AutoDebug completed (AUTODEBUG.md): verified rank3/rank4 distributed-norm bug;
packed ownership and local KV/conv/native-prefill indexing cleared by inspection.
AutoFix applied rank adapter and exact local12 optimized recurrence adaptation;
AUTOFIX.md records checks and pending runtime verification. Unchanged optimized
compute/writer retain rounding/readout and queue contracts. Host checks pass630
partitions and seven rank adapter shapes; static runtime audit now29 functions.
Runner baseline-source provenance corrected. No further device work or resets.
No stage review or local checkpoint commit yet: hardware gates are incomplete.

AutoDebug exited0 after final static inspection of the applied local12 changes;
its report records2016 partition/QK lifetime checks. Full launcher output is
compressed in autodebug_launch_retry.log.gz. Reset authorization is still pending.

Continuation 2 (host-only; reset authorization still pending):
- Revalidated worktree and prior evidence. Previous turn classified as progress:
  implementation, AutoDebug/AutoFix and host-contract evidence were produced.
- Strengthened trace harness with changed per-row positions/RoPE and physical
  page remapping. Remapping moves physical cache pages and IDs together, so
  the logical history must remain bit-identical. Eager and replay output/state
  must agree exactly, with per-case PCC against the optimized single-chip run.
- Replays now assert determinism instead of merely recording a Boolean.
- References carry layer/batch/length, pinned HF revision, baseline-source hash
  and input/page-table hashes. Old baseline_l0.pt/json must be regenerated.
- Added host validation of the actual remapping helper:32 configurations over
  B1/2/3/32, pages1/2/16/17, local KV-head counts1/4, four distinct cache shards
  and unused physical IDs. Logical histories agree; changing IDs without
  moving data fails the negative control. host_validation.json also retains
  the630 ownership-partition and seven distributed-norm rank checks.
- Commands: pre-commit on run_multichip_decoder.py/validate_multichip_host.py
  (hooks_trace_cases.log); python tests/validate_multichip_host.py;
  python -m compileall -q tests/run_multichip_decoder.py, with paths relative
  to this model directory. All host checks passed. No TTNN import/open/reset,
  device run, stage review or commit occurred in this continuation.
- Device validation remains the next discriminating action, dependent on the
  unanswered reset authorization question. No hardware gate is claimed passed.

Continuation 3 — blocked audit:
- Previous turn classified as progress: trace-case/provenance changes and32
  host remapping checks changed the authoritative test implementation/evidence.
- Rechecked git status, host_validation.json, all result JSON inventory and the
  installed device-usage reset rule. Still no multichip forward result exists.
- Same blocker across three goal turns: explicit reset authorization is absent.
  Original turn applied source repairs; continuation2 strengthened host tests;
  continuation3 has reached the point where target-device evidence is required.
  The automated continuation instruction is not an answer to the pending
  authorization question. No reset authorization is inferred from elapsed time.
- The failed multi-chip setup run closed normally; this is not a claim that
  hardware is wedged. The selected multichip protocol requires reset after a
  failed multi-chip run, while device-usage reserves resets for operator
  authorization. Bypassing that protocol, switching to a smaller mesh, or
  accepting host-only checks would not satisfy the requested stage.
- Goal status: blocked pending operator authorization. AutoFix source repairs
  passed host checks; device verification is unavailable under the current
  authorization. This is not an AutoFix success/failure claim on hardware.
- Concrete next action after authorization: check fuser for all four devices;
  if idle, bounded tt-smi reset/list and1x4 FABRIC_1D mesh smoke; regenerate
  optimized reference with current provenance, then resume TP4 correctness,
  trace/cache/context, topology/performance, watcher, review and local commits.
- All source and evidence remain on persistent storage. No reset, stage-review,
  checkpoint commit, push, full-model work or vLLM work was performed.

## Owner reset authorization and resume (2026-11-13)

Owner explicitly authorizes `tt-smi -r` on idle devices or after terminating
our own hung test process. Goal resumed; fuser found no owners on devices0–3.
Running bounded reset/list/1x4 FABRIC_1D smoke before correctness tests.
No external processes killed and no locks removed.

Reset completed exit0; post_reset_inventory.log lists all4 P300c devices.
post_reset_mesh_smoke.log records MESH_SMOKE_OK, exit0. No second reset needed.

`run_job.sh baseline_l0_v2 python -m models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder --baseline --layer 0 --batch 1 --length 33 --reference doc/multichip_decoder/ref_l0.pt --output doc/multichip_decoder/baseline_l0_v2.json` (model-relative artifact paths expanded in actual command) passes.
Same command without --baseline, tag tp4_l0_v3, passes on all4devices: prefill0.9999821820, decode0.9999763831, contribution minimum0.9999762962, state minimum0.9999797605.
This is the first actual TP4 forward validation; trace and remaining gates pending.

## Cold B32 repair validated

Shared manager allocated semaphores only on8x8; CCL default worker traversal
uses Blackhole11x10. Local _BlackholeCCLManager covers fullgrid. Initial B32
NaN reproduced both regular and watcher cold runs before repair. After repair,
fresh-cache watcher_cold_b32_v2 passes with0/476 JIT cache hits; no watcher
errors. Exact original/regression commands preserved in logs and suites.
Correctness_v2 covers both kinds B3/S513,B1/S31,B1/S1025; correctness_v3
covers full attention B2/S33 andB32/S33 including trace/positions/page remaps.

## Topology experiments

MMRS MLP S512 with next sharded residual/norm consumer passes PCC0.9999929;
warmed trace boundary0.500857ms separate versus0.406864ms fused.
Legacy fused AGMM rejects Linear (native explicit invariant). Adapted to
all_gather_minimal_matmul_async; first config rejected sender-axis count.
Adjusted8x8 grid to2links x4workers/link. This compiled but stalled.
Captured full live tt-triage (triage_agminimal/) before terminating only
our own PID2070747 with SIGKILL after184s, exit137; no external process touched.
Owner reset authorization covers recovery; reset_agminimal_hang.log records
bounded reset/list/mesh smoke. AutoTriage subagent inspects saved evidence.
Model-local flat CCL/norm and fused-MMRS prefill candidates prepared;
padding stays internal and public logical sequence length is preserved.

## Profile-driven batch packing (11:25–11:30 UTC)

`profile_initial_l0_b32` is invalid: Tracy's shell reconstruction stripped JSON
quotes before decoder setup; wrapper exit0 did not prove target success. Added
`--policy-file` and reran `profile_initial_l0_b32_v2`; JSON and report both exist.
`perf/profile_initial_l0_b32_v2/` contains advice-enabled human tables, CSV,
compressed raw ops, metrics/provenance and accounting. Same-run host2.8416ms,
device kernels2.7455ms, kernels+gaps2.8540ms (adjacent replay measurement
variation explains -0.0123ms gap). Conv32 launches594us; reshapes727us;
matmuls356us; collective94us. This directs work at batch padding/convolution,
not speculative fabric bandwidth changes.

Initial local batched convolution omitted `packed_qkv=True` at the specialized
recurrence call and failed a Python shape assertion (`batched_conv_v1_l0_b32`).
Corrected the caller, added compact whole-decode batch flow, and reset devices
(`reset_batched_conv_assert.log`). The synchronized topology probe had already
been launched after the assertion and completed before reset; it is not used
as a final clean-state validation. `flat_decode_v1_l0_b32` passes: output
PCC0.9999632, state minimum0.9999786, trace bit-equal, warmed0.92592ms.
The public residual remains[B,1,1280]; internal[1,B,C] avoids32 physical rows
per logical decode user. Batched convolution uses FP32 products/accumulation,
BF16 rounded sum then SILU, matching baseline arithmetic policy.

AGminimal AutoFix: persistent5MiB gather scratch and dedicated full-grid
semaphores pass eager and50 synchronized trace replays (PCC0.99999999997).
`topology_agminimal_persistent_sync.json` timings include host fences and are
not device performance. Removing only replay fences hangs after successful
first replay; live capture is `triage_agminimal_async/`. This isolates repeated
launch admission as remaining risk. Native Linear fused-gather lacks entry
barrier; ordinary AG's minimal_default_writer.cpp:258–310 implements one.
Next bounded experiment inserts a tiny persistent AG before the fused op,
including its cost in the traced candidate.

## Fusion and compaction decisions

`topology_agminimal_best` reruns the repaired, device-fenced fused op against
the actual current8x8 separate projection, not the older8x4 control. Both are
correct, but separate0.339615ms beats fused0.431357ms. Reject fused gather on
measured performance; retain its repaired probe and triage evidence. Fused
row-matmul/reduce-scatter remains a kept prefill candidate.

`compaction_local` passes exact per-user/rank checks for B2,3,9,10,17,31,32:
four distinct tokens/history shifts; compact vs expanded convolution; user
permutations; compact gate reshape; packed vs independent separate-RM Q/K/V
recurrence; resulting output and persistent state. No aggregate-PCC-only mask.

`candidate_v2` completes16 same-harness whole-layer jobs. B32 GDN compact
improves2.748→0.926ms; B32 full attention2.086→1.612ms. Sharded residual retains
substantial prefill gains versus replicated; replicated wins someB1 decode
rows, so gather-before-normalization is a pending contract alternative.
Full-attention compact head/projection candidate `flat_attention_v1_l3_b32`
passes all per-user PCC, cache/position/remap and trace checks,1.515ms.
The next measurements compare DRAM reader/core geometry and normalization.

## Integrated local-head implementation and precision search

`candidate_v3_commands.jsonl` records24 jobs: baseline, current, gathered
normalization, packed MLP, native prefill SwiGLU and SDPA64. Selected the
phase-specific norm contract (GDN gather-before-norm decode; attention
single-user gather-before-norm), packed gate/up decode and tile-pair-packed
native SwiGLU prefill. Independent SwiGLU and row-MMRS copies are counted in
`capacity_plan.json`; separate gate/up allocations are removed at setup.

`profile_initial_l3_b32` shows325 decode device ops and633us of tensor movement.
The model-local compact QKV staging specializes the optimized baseline to
Q6/K1/V1. It zeros complete512-byte RISC-owned faces, then scatters live rows;
K/V update shards are disjoint, and concat's physical32 users are cropped to
logical B. `compact_attention_v1_l3_b32` passes PCC/cache/remapping and
allocation-tracked replay at0.7181ms. `qkv_staging.json` checks exact per-rank,
per-user Q/K/V values and padded zeros for B1,3,9,17,31,32; changed and permuted
inputs reuse compiled programs. All24 cases pass.

`integrated_v1_commands.jsonl` validates selected defaults against fresh
optimized references for both kinds at B1/S512, B32/S33 and B3/S513. All
prefill/decode/state/per-user/contribution PCC gates and tracked trace cases
pass. These are intermediate correctness numbers before precision selection.

`fidelity_v1_commands.jsonl` uses recorded target activations, identified by
SHA in each result. GDN attention HiFi2 improves0.8140→0.7657ms; LoFi0.7640ms.
MLP LoFi and BFP8 CCL produce no material decode gain. Full attention already
uses LoFi; raising fidelity adds prefill cost. `geometry_precision_v1` extends
real activations to B32/S254 and tests K4/8/16 plus four activation workers.
LoFi improves prefill52.55→49.99ms vs HiFi2 and passes state PCC>0.99952;
further same-policy geometry and BFP4 stage-8 candidates precede final selection.

Updated capacity includes every selected matrix copy, padded norm/control
vectors, DRAM bank rounding, all64 layers, conservative BF16 embedding/head,
all16 attention caches at context262144 and all48 GDN states. Resident estimate
16,635,420,672bytes/device; with4GiB transient allowance20,930,387,968bytes.
The planned on-device context gate reserves20GiB in addition to the actual
layer/cache/IO, exceeding the estimated complete resident stack. It tests
aggregate capacity, not identical future full-model fragmentation.

LoFi geometry repeats at B32/S33 retain K2: GDN0.76409ms versus K8 0.77628,
K16 0.78950, cores4 0.76643; full0.70166 versus0.70643/0.71642/0.70370.
Selected LoFi attention after real-activation S33/S254 comparison. MLP stays
baseline fidelity; BF16 CCL remains selected. Source formatting precedes final
measurements; subsequent artifacts record the default policy and physical
matrix allocations as well as hashes.

BFP4 attention initially hit native tile-pad dtype validation. Adapted setup
through exactly representable BF16, then repacked BFP4; reset and reran both
kinds (`bfp4_v2`). MLP BFP4/LoFi and GDN attention BFP4/LoFi pass and are stage8
candidates, not defaults under the optimize skill. Full-attention BFP4 gives
KV PCC0.9937779/0.9946542 on real recorded activations, below0.995; reject that
candidate for cache accuracy. Output itself0.9998875 and replay bit-equal:
this is quantization accuracy, not a cache indexing or trace bug. Defaults
remain BFP8. Reset/mesh smoke evidence: reset_bfp4*.log.

Source-only resource audit found64 independent layers use912 global semaphore
objects (~58,368bytes/L1 bank), plus2,048bytes of shared MMRS resources. Added a
64-instance shared-weight resource probe to check actual L1/cache/trace capacity.
Context harness now checks untouched sampled cache rows after continuation,
continuation vs original tail, and traced state writes with changed input.
PCC oracles fail closed on non-finite/undefined correlations, with exact-equal
constant tensors explicitly accepted. Source audit36 reachable functions,
zero runtime host-conversion violations.

### Maximum-context, sustained decode, and real stacked boundaries

The final-context jobs at source `543837e9169846679901dc81d9450e01619ef2d333307f5442a80ae36cb4203d`
passed both layer kinds at 262143 and 262144 logical tokens with an additional
20GiB DRAM reservation on every device. No context reduction is needed.
Artifacts: `final_context_l{0,3}_b1_s{262143,262144}_{baseline,tp4}.{json,log,command.json}`.
GDN minimum selected contribution PCC was 0.997033; full-attention minimum
selected contribution PCC was 0.9999729. Maximum measured allocated DRAM was
23,274,733,568 bytes/device, including the reservation. First full-attention
maximum-length execution included JIT compilation and is not a warmed latency.

`final_stress_l{0,3}_b{1,32}_s{128,33}_{baseline,tp4}` passed 128 steps at B1
and 512 steps at B32, with changed inputs, advancing full-attention positions,
and repeat-state determinism. Lowest final GDN state PCC was 0.9965569.

The earlier synthetic chain `[0,3,0]` at B2/S65 failed decode PCC (0.9931346)
under selected attention LoFi. A matched higher-fidelity control passed
(prefill 0.9998173, decode 0.9999254):
`stack_synthetic_control_{baseline,hifi4}`. This demonstrates precision
accumulation, rather than permission to weaken the threshold. The real chain
of consecutive checkpoint layers `[0,1,2,3,4]`, using recorded layer-0 model
activations and its actual next token, passed selected LoFi with no boundary
conversion and bit-equal trace replay: B2/S65 PCC 0.9999443/0.9999575 and
B32/S33 PCC 0.9999387/0.9999712. Artifacts:
`stack_real_{baseline,lofi}_b{2,32}`. The stack orchestration now uses that
recorded-activation, consecutive-layer fixture. This is a decoder interface
check, not end-to-end model quality validation.

### Narrow correction and final 64-owner resource gates

Restored the baseline's narrow-GDN correction optimization with local TP4
widths, after a four-way controlled experiment (`narrow_v1_*`). B1/S512 and
B32/S33 use recorded target-model activations. Narrow+L1 wins decode at both
batches (0.57404/0.73298ms versus wide+DRAM0.60360/0.76418ms); narrow-only
0.58259/0.74344ms. Every candidate passes output/contribution/state PCC and
trace checks. The production setup removes zero Z columns before BFP8 upload;
all retained columns are tile aligned. The native combine specializes the
optimized baseline's kernel, asserting expected source-token substitutions.
Production source after formatting:
`384f75b273279e13b64327bc5aeef3612244997f7ac1804c78131220b386155c`.
The historical probe explicitly forces a wide parent before applying its own
candidate, preserving reproducibility after default selection.

`final_resources_b1.json` and `final_resources_b32.json` both pass with
`TT_METAL_WATCHER=10`, allocation tracking1 and program-cache tracking included.
All64 independent owners retain768 CCL semaphores,144 convolution semaphores,
and48 exchange buffers. Semaphores plus native-MMRS resources occupy60,416
bytes per L1 bank (1,399,872 bytes/bank still free before capture). Main trace
allocations are17,104,896 bytes at B1 and29,818,880 at B32, within the32MiB
region. Original→changed→original replays compare every complete output and
mutable state tensor on all four ranks, exactly. Program-cache growth after
warming is zero. Watcher log copies accompany each result. Shared readonly
weights mean this is a resource/lifetime gate, not a full-model accuracy test.

Capacity after narrowing: estimated resident16,234,340,352 bytes/device,
including all64 layer matrices/copies, BF16 embedding/head estimates, full
262144 KV caches and padded auxiliary tensors. With4GiB decoder working
reserve, total20,529,307,648 bytes/device. The separate20GiB test allocation
alone exceeds the entire estimated resident set by5,240,496,128 bytes.

The final packed-SwiGLU topology comparison is `final_packed_agminimal`:
separate AG+native SwiGLU0.496683ms, fused AG-matmul+SILU/multiply+admission
fence0.753429ms. Fifty queued replays pass PCC0.99999267, with persistent
scratch and dedicated semaphores. This coherent final packing/fidelity/layout
comparison rejects the fused path on measured latency, not its earlier API or
admission failures.

Ran isort5.13.2, autoflake2.3.1 and Black on stage Python files. Production
changed only import ordering (exact diff and before/after hashes in
`format_provenance.json`), final SHA
`de734a896dc879ef5e4d3ebfb2f41f4734d53e31af1cae52c48d2d8775e42f3e`.
The correctness suite source-freeze guard caught this change between successful
B2/S1 jobs and stopped before the next device job. Preserved those artifacts as
`preformat_correctness*`; restarted the entire correctness suite after formatting.
This was a host provenance guard, not a hardware or numerical failure. Final
performance/profiles and renewed context/stress/stack jobs run the frozen source.
Resource and topology evidence uses the immediately preceding source, differing
only by the recorded import-order change.

### Final paired latency

`final_perf_{baseline,sharded,replicated}_l{0,3}_b*_s*` completes all18 jobs
with watcher/profiler/allocation tracking disabled for timing and100 queued
warmed trace replays. Output/cache PCC and trace equality are still checked.
`performance.{md,csv,json}` contains six paired sharded rows, speedup and
four-chip efficiency for both phases. GDN decode1.158→0.574ms(B1),
1.671→0.733ms(B32); full attention1.022→0.490ms(B1),1.173→0.702ms(B32/S33),
1.283→0.749ms(B32/S513). Prefill speedup ranges1.32–3.21x across the six rows.
The report generator initially included `.command.json` sidecars in its glob;
fixed the host-only filter and regenerated from untouched measured artifacts.

`residual_layout_comparison.json` records the final control. Sharding wins all
prefill rows and B32 decode, with a roughly3% B1 decode tradeoff versus
replication. The selected layout targets the largest batch and lowers residual
memory4x; final numbers reflect that choice, not a mixture of best-per-row
contracts. Baseline source remains unchanged.

### Watcher gate reopens fused prefill acceptance

The final watcher GDN B32/S33 job aborts in native
`matmul_reduce_scatter_prefill_writer.cpp` at kernel completion with pending
NoC reads. A second fresh-device reproduction (`mmrs_watcher_repro`) fails
identically. Numerical/performance passes do not waive this gate. Preserved
both watcher logs and kernel maps before idle-device resets. Post-crash
`tt-triage` could not attach to the already-closed inspector; its summary is
not live hang evidence. A fresh source-only AutoDebug child is investigating
in `debug_narrow_watcher/`; model-local candidates will retain all watcher
checks. Shared native C++ is outside this stage's authorized edit scope.

The final profiler also identifies ordinary prefill matmul and native SwiGLU
configuration follow-ups. Those sweeps remain open; final acceptance and
source provenance will be refreshed after the watcher repair and any selected
configuration changes.

Watcher discrimination: separate matmul/RS (`mmrs_disabled_watcher`) and
128-row fused chunks (`mmrs_chunk128_watcher_v2`) both pass B32/S33 GDN,
including changed-input trace replay and allocation positive control. Native
last-plan records confirm resident M-on-Y grids4x10 and1x10 with NoC0 reads
for both attention and MLP chunks. Uninstrumented matched measurements are
12.8747ms separate versus11.5321ms chunked prefill (unsafe original10.5250ms).
Decode remains0.733ms. The first chunk wrapper exited cleanly because the
private plan getter belongs under `_ttnn.operations.experimental.ccl_experimental`,
not the public `ttnn.experimental` namespace; fixed only the test wrapper.

The ordinary prefill K16 candidate hit a host divisibility assertion at MLP
down K4352=136 tiles:16 does not divide136. Preserved `invalid_global_k16*`.
The resumed candidate uses K16 on divisible projections and K8 on MLP down;
this is a corrected experiment configuration, not a numerical rejection.

### Safe prefill configuration comparison

`safe_prefill_geometry_summary.{csv,json}` records all28 ordinary-projection
jobs. All numerical/trace gates pass after correcting the invalid global K16
candidate. At B1/S512, default8x8/K4 remains competitive; at B32/S254,
11x10/K8 is faster (GDN60.413→56.168ms; full47.522→46.056ms), with some
small-shape tradeoff. Final integration will check the coupled choice rather
than assuming this separate-row-path gain transfers unchanged.

`native_prefill_candidates_summary.{csv,json}` compares native API arguments
on the same real activation references, keeping the original oracle. Separate
row matmul/RS wins B1/S512 (GDN2.278vs2.810ms chunked; full1.784vs1.981).
At B32/S254,128-row fused pieces win (GDN60.960→54.967; full47.219→42.611).
This selects separate row projections at flattened M<=512 and safe128-row
fusion above512. Decode continues its original row-parallel path. Native
NoC1 behavior is never intentionally selected by the new final path.

Small-shape native SwiGLU config2,8,26,1,2,11,8 beats the exact-partition
64-core alternative: full1.784→1.691ms, GDN2.278→2.242. Use it for33..512
rows; keep the existing110-core config for larger shapes and one-tile tails.
The large-shape K8/N26 alternatives offer no material consistent improvement.
The GDN tail-specific22-core candidate is slightly slower67.559→67.612ms;
no aligned-only public contract or extra logical padding is introduced.

Integrated coupled geometry checks (`safe_integrated_geometry*`) all pass.
Select11x10/K8: GDN B32/S25454.652→51.908ms, full42.790→42.366ms;
full B1/S5121.740→1.640ms, GDN2.151→2.245ms. This favors large-prompt
throughput with a recorded0.094ms GDN small-prompt cost, not a claim that one
configuration wins every shape. Decode geometry remains unchanged. Formatting
and runtime/host contract audits are completed before the final source freeze.
The new `run_safe_gates.sh` serializes all accepted-source gates under
`final_safe_*`, leaving earlier rejected-source evidence intact.

### Accepted-source resource gate

Frozen production SHA256
`63d2e53af8490dd63ec081ea202611e21dfb3ddca3e033e851e89837d0debb88`.
`final_safe_resources_b1` passes B1/S65 prefill and B1 decode;
`final_safe_resources_b32` passes actual B32/S33 prefill and B32 decode.
Both watcher10/allocation-tracking runs retain64 distinct CCL owners,
48 convolution resource sets, exact original/changed/original output and
state hashes, zero late program-cache growth, and unsafe-allocation positive
controls. Trace allocations remain17,104,896 and29,818,880 bytes/device within
32MiB. This closes the focused native-workaround L1/resource-pressure concern;
whole-layer awkward-shape watcher and numerical suites follow in the same
serialized `run_safe_gates.sh` session. No source changes occur during it.

### Large-prefill crossover discovered by final latency gate

All final_safe resource, watcher (six TP4 cases), correctness (ten TP4 cases),
paired performance (18 jobs), and four profiler jobs pass on63d2e53a. Before
stress/context/stack began, the final B32/S513 comparison showed that chunked
fusion52.531ms/full attention is materially slower than the previously measured
separate path46.905ms. The orchestrator was suspended while its profile child
finished, then terminated after the child closed and `fuser` showed no device
owners. No device test was interrupted and no reset was needed. This exit143
is an intentional host orchestration stop, not a failed device gate.

`safe_large_separate_l{0,3}` now compares the separate path under the exact
selected11x10/K8 geometry, small-SwiGLU policy and same final_safe references.
A measured crossover, if confirmed, will be integrated before the delivery
source freeze. Existing final_safe artifacts remain intact as intermediate
safe-source evidence; no final acceptance claim uses their source hash after
a subsequent implementation change.

The exact-policy large-prefill control passes PCC/cache/trace and confirms the
crossover: GDN67.347→59.182ms, full52.531→44.758ms at B32/S513.
The final row dispatch uses native128-row pieces only for512<M<=8192;
otherwise it uses separate matmul/RS. Thus no mathematical padding or public
sequence-length restriction is added. Added watcher B17/S33 and B31/S31 to
exercise native64/96-row tails, in addition to32/128-row cases; profiles now
include B32/S513 for the large separate path. Static audit38functions/0violations
and630 host partition cases pass after formatting.

Delivery source SHA256:
`082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44`.
`run_delivery_gates.sh` runs the complete serial source-frozen battery under
`delivery_*`, including the focused compaction watcher test. Earlier final_safe
and original final artifacts are historical controls and remain preserved.

### B31/S31 synthetic precision sensitivity

The added96-row-tail watcher case completed prefill/decode without a device
assertion, but random-input decode user25 scored0.994696745 against0.995.
The initial command exited before saving numerical details. The runner now
serializes failed initial PCC reports before raising; the same per-user,
global, contribution and state0.995 bars remain in force. Exact runner hashes
are in `initial_pcc_failure_reporting.json`. Production stays082c82fa.

Fresh source-only AutoDebug runs in `debug_pcc31/`. Matched controls on disk:
`pcc31_default_synthetic` reproduces; `pcc31_separate_synthetic` has identical
per-user decode and state PCCs, eliminating the prefill fused-tail path as the
cause of this decode discrepancy. `pcc31_hifi4_synthetic` changes only attention
fidelity and passes (minimum per-user decode0.999210842). Default LoFi on
recorded target-model activations passes both kinds at the exact B31/S31 shape:
GDN minima prefill0.999979012/decode0.999991987; full0.999989163/0.999996138,
with watcher and exact changed-input trace replay. `pcc31_control_summary.json`
records the comparisons without treating equal PCC scores as tensor hashes.

The fixture is a deterministic slice of the prior-stage recorded B32/S33
activations: users0..30, prefill tokens0..30, decode token31. The source and
slice hashes are in `text_l{0,3}_b31_s31.json`; `prepare_pcc31_inputs.py`
recreates it. This preserves the0.995 acceptance bar on actual target inputs.
Per the optimization precision rule, a random-activation sensitivity alone
is not used to select a slower fidelity when recorded target inputs pass.
The synthetic counterexample remains a disclosed accuracy limitation and is
not claimed as a pass; full-model accuracy is a later-stage gate.

Initial failed artifacts were preserved as `pcc31_initial_{baseline,sharded}*`
and the old reference as `pcc31_synthetic_ref.pt`; rename hashes are in
`debug_pcc31/ARTIFACT_RENAMES.json`. The delivery watcher suite now explicitly
uses recorded inputs for31x31, with unchanged synthetic inputs for its other
cases. `resume_delivery_gates.sh` reruns that pair and the remaining full-
attention watcher cases, then continues all pending gates. Previously passed
082c82fa resource and GDN watcher cases are retained. No reset or device-process
termination was needed for this numerical failure.

### Delivery performance and profiler recovery

All ten delivery correctness cases, ten watcher cases, both 64-owner resource
runs and eighteen baseline/sharded/replicated performance runs passed on source
082c82fa. `performance.md/.csv/.json` and `residual_layout_comparison.json` now
use these final-source measurements. TP4 decode speedup is 1.67–2.28x;
prefill speedup is 1.22–3.14x over the frozen optimized baseline. The sharded
residual wins every measured prefill and B32 decode; replicated B1 decode is
slightly faster, with a larger residual footprint.

The resumed chain stopped during the second profiler run. The missing CSV was
a downstream error: profiler.cpp aborted on an unmatched `pp_intra` end marker
in ChunkGdnDeviceOperation, TRISC0, chip3/core(12,11), during report processing after both timed prefill
and decode signposts. Evidence and exact argv are preserved under
`debug_profiler_marker/`; a fresh source-only AutoDebug investigation was
launched. No production change or accepted profile claim was made. Devices
were idle after the aborted process; the authorized bounded reset and mesh
smoke passed (`debug_profiler_marker/reset.log`), without killing any process
or clearing locks. `finish_delivery_validation.sh` continues the independent
stress/context/stack/layout gates, serially, while the host investigation runs.

AutoDebug identifies an incoherent legacy Blackhole clock read: live high word
is read before low; the observed orphan timestamp has low word 10 and can be
backdated by 2^32 cycles across rollover. This execution-level attribution is
an inference because raw neighboring device markers were not emitted. Parent
verified the high/low source order and that MID_RUN_DUMP skips the failing
pairing check; that switch is rejected. The minimal collection repair is a
fresh process with identical settings and normal pairing validation. Accepted
profiles will use new `delivery_profile_retry_*` paths. Existing failed logs,
command/JSON and host Tracy files remain in `debug_profiler_marker/` (raw host
capture under `tracy/failed_l0_b32/`). No production or shared profiler source
is changed. The permanent shared clock-sampler repair is outside this goal.

### Delivery stress, maximum context and stacked handoff

All four sustained-decode runs pass and are bit-exact across two repetitions:
GDN B1/B32 output PCC 0.999777249/0.999537998; full attention
0.999924070/0.999816288. B1 runs128 steps and B32 runs512; full-attention
positions advance across physical pages.

Both layer kinds pass262143/262144 on the delivered source, executing every
prefill token with an additional20GiB/device reservation. GDN maximum prefill
PCC is0.999497470 and last-legal decode0.999654682; full attention
0.999988033/0.999990241. Minimum selected contribution PCC at max is
0.997289198/0.999970333. Sampled per-device allocation peaks are
23,043,613,184 and23,278,714,880 bytes, including reservation; concatenation
can add about671MB transiently. `finalize_context.py` refreshes only the
multichip nested contract and capacity report, preserving all earlier stages.
Supported context remains262144 without reduction. Estimated all-stack
resident bytes are16,234,340,352, plus4GiB transient reserve; phase copies,
padded auxiliary vectors, all local KV caches and recurrent states are counted.

The actual five consecutive checkpoint-layer stack passes without boundary
conversions at B2/S65 and B32/S33: prefill/decode PCC
0.999944546/0.999957129 and0.999940026/0.999965810 respectively. The stack trace compares its final original-input output to eager exactly;
changed-input and complete state equality belong to the separate single-layer
and64-owner resource tests. This is layer-stack
handoff evidence, not a full-model implementation or quality claim.

The delivered64-owner resource runs leave58,368 bytes/L1 bank allocated at B1,
and60,416 at B32 after native MMRS warmup; largest contiguous free regions are
1,401,920 and1,399,872 bytes/bank. Trace allocations are17,104,896 and29,818,880
bytes/device, within the32MiB region. There is no late program-cache growth.
The focused QKV and compaction watcher tests pass, and the ordinary pytest
entrypoint reports4 passed in65.28s. All device validation has closed normally;
only the serialized profiler recollection/report chain remains before review.

### Final profiles and aggregate gate

All six `delivery_profile_retry_*` device/Tracy jobs exited0 (34,68,210,27,47,
116 seconds). Normal pairing checks remain enabled, no marker-drop/error
warning appears, all four ranks are present, and raw ops hashes match metrics.
The aggregate verifies actual BFP8 matmul weights and LoFi/HiFi2 policy rows.
DRAM projections reach293–443GB/s of512GB/s nominal. Decode explicit CCL is
76–96us; kernel sums are553/716–717us GDN and463/675–723us full attention.
B32/S33 prefill physical reshapes total4.043/3.686ms and remain a bottleneck;
large prefill has12.3–12.5ms explicit CCL. Native MMRS embeds communication in
Other, explicitly separated in the report. All six human tables, CSV, raw ops,
commands and accounting files exist. Profiler timings are not used for speedup.

The profile driver itself exited127 after finishing the six jobs and all
reports: editing its early `unset` line while Bash was reading the script
shifted the file offset, leaving a stray `f` command at EOF. This was a parent
orchestration mistake, not a successful whole-driver exit. `bash -n` now passes;
all six independent wait_for records are0. Re-running all six host accounting
steps, the report renderer and aggregate gate exits0. No device job was cut
short, no report is missing, and no rerun was required for numerical evidence.
The wrapper is now frozen; future edits occur only after its process closes.

`gate_summary.json` passes:10 correctness,6 selected performance,10 watcher,
4 stress,4 full-context,2 mixed-stack,2 resource,6 profile and2 local-kernel
artifacts, plus4 pytest cases, host contracts and clean runtime audit. Synthetic
precision controls remain disclosed separately. Independent stage review and
local checkpoint commits are the remaining closure steps.

Repository pre-commit checks pass (`precommit_delivery_final.log`). The first
run normalized only trailing whitespace/final blank lines in human-readable
reports and triage summaries; numeric CSV/raw compressed evidence is unchanged.
The archive has649 entries, each verified by stored and decompressed SHA256
against its original. Raw PT/Tracy and private agent transcripts stay local;
compact reports and lossless required logs are staged. The delivered decoder
source remains082c82fa. All measurements and checks are closed before review.

### Independent stage review, round1

`stage_review_round1.md` returns more-work-needed: P2 coverage at neighboring
logical lengths around the final512/8192-row dispatch thresholds. It found
no implementation defect or numerical contradiction in accepted evidence.
The requested additions are both layer kinds at B2/S256,S257 and B32/S256,S257,
with the unchanged PCC bars, watcher and trace-allocation controls.

`run_multichip_boundaries.py` wraps only the test harness: it observes the
first eager prefill's two row projections, recording real input/output shapes
and calls to native MMRS. No tensors are converted and no device operation is
added; patches forward the original operation and are not performance evidence.
The production decoder and core numerical runner remain unchanged. The serial
command is `python -m models.autoports.qwen_qwen3_6_27b.tests.multichip_jobs
--suite watcher --tag delivery_boundary --shapes 2x256,2x257,32x256,32x257
--record-row-dispatch` (the command JSONL retains exact argv with spaces).
The aggregate now requires these eight cases, both sibling row consumers and
observed512/514/8192/8224-row paths, as well as direct finite >=0.995 numerical
checks for all ordinary accepted cases.

The review's documentation concerns are corrected: current prefill fusion is
512<M<=8192; the real stack is checkpoint layers0..4. Its one original-input
trace compares final output only. Changed-input and complete state equality
are attributed to the separate single-layer and64-owner resource tests.
The measured reshape cost and replicated B1 advantage remain next-stage
optimization opportunities, not claims that those costs are eliminated.

All eight requested boundary runs pass with watcher10, trace-allocation tracking
including program-cache allocations and a passing allocation-positive control.
The observed native piece counts are0/5/64/0 for512/514/8192/8224 rows,
respectively, for both attention output and MLP down projection.

- `delivery_boundary_sharded_l0_b2_s256.json`: minimum PCC 0.999174291; rows 512, both projections `separate_matmul`.
- `delivery_boundary_sharded_l0_b2_s257.json`: minimum PCC 0.999159352; rows 514, both projections `native_mmrs`.
- `delivery_boundary_sharded_l0_b32_s256.json`: minimum PCC 0.996169386; rows 8192, both projections `native_mmrs`.
- `delivery_boundary_sharded_l0_b32_s257.json`: minimum PCC 0.998257820; rows 8224, both projections `separate_matmul`.
- `delivery_boundary_sharded_l3_b2_s256.json`: minimum PCC 0.999644911; rows 512, both projections `separate_matmul`.
- `delivery_boundary_sharded_l3_b2_s257.json`: minimum PCC 0.999665039; rows 514, both projections `native_mmrs`.
- `delivery_boundary_sharded_l3_b32_s256.json`: minimum PCC 0.999473009; rows 8192, both projections `native_mmrs`.
- `delivery_boundary_sharded_l3_b32_s257.json`: minimum PCC 0.999210837; rows 8224, both projections `separate_matmul`.

The host aggregate exits0 with54 accepted artifacts:8 boundary,10 correctness,
6 performance,10 watcher,4 stress,4 context,2 stack,2 resources,6 profile,
and2 local kernels. Command: `python -m models.autoports.qwen_qwen3_6_27b.tests.collect_multichip_evidence --prefix delivery --profile-prefix delivery_profile_retry`.

Final packaging has684 lossless archive entries, verified against stored and
original SHA256/length. A self-check caught the private transcript exclusion
preceding the general compressed-log include rule; the packer now emits the
private exclusions last, and the private transcript is removed from staging.
No device artifact or implementation changed. Pre-commit passes on the packer
and ignore-rule correction; all added/modified staged paths are stage-owned
model decoder/tests/docs, with no PT, Tracy database or private transcript.

### Independent stage review, round2 and closure

The fresh xhigh diff-only reviewer returns **clean-pass** in
`stage_review_round2.md`. Every prior finding is closed; no Required Work
remains. The reviewer independently checked336 initial numerical scalars,
144 trace comparison scalars, both observed row consumers, paired baseline
provenance, watcher/trace checks and all35 new archived records. The existing
implementation, capacity, stress and performance evidence remains valid.
Boundary startup topology/clock and semaphore-fragmentation advisories are
classified as controlled in that report; they do not enter performance claims.

Stage scope is the TP4 decoder plus its tests/docs/context plan. The optimized
baseline is unchanged. No full-model or vLLM work was started. Local checkpoint
commits follow this passing review; no push is authorized or performed.

Local stage checkpoint: repo `/home/ttuser/qwen36-27b/tt-metal`, branch `main`,
commit `79737a5e44d090d0e5d71b7e01ec5cb95777e197`
(`Add Qwen3.6-27B TP4 decoder with validated mesh evidence`). Final pre-commit
and CRLF-aware staged whitespace checks pass. The checkpoint includes the
clean-pass review and all stage-owned implementation/tests/docs/evidence.
The worktree was clean immediately after the checkpoint. This SHA record is
committed in a following documentation-only commit. No repository was pushed.
