# AutoFix: sliding BFP8 attention CCL replay

Status (updated checkpoint): first changing logical value localized to the
indexed expert gate slice on rank 1. The active unresolved region is the
gate/up sparse projection or its slice, before GELU/down/mix. Original replay
failures persist; no production fix is retained. Exact equality is unchanged.
Hardware is released to the coordinator after both frozen-prefix controls
closed cleanly. Earlier ownership/release notes below are historical.
Router placement(1,0) is a promising model-local candidate: original-path128
and1024 duplicate-check stress pass, while a restored core0 control fails.
The coordinator has integrated the model-local placement workaround; ordinary
accuracy, replay and adjacent acceptance gates are pending.

The authoritative detailed record is [AUTODEBUG_bfp8_ccl.md](AUTODEBUG_bfp8_ccl.md).
Its initial source-only report preceded authorized diagnostic/hardware work.
V1/V2 retained runs used the wrong unsharded normalization override and cannot
prove equivalence or a retention fix. V3 and later use the actual width-sharded
`OptimizedDecoder.normalize` sequence. Physical comparisons also distinguish
stable NaN bits and continue harmless padding-only changes while retaining
those statistics; logical NaN and replay checks remain strict.

## Current experiment ladder

| Experiment / artifact | Result | What it establishes |
| --- | --- | --- |
| Original output-only `bfp8_output_v2` | Fails step 1, finite replicated variation | Current runtime reproducer |
| Corrected outer boundaries `bfp8_boundary_v3_all` | Fails step 6; first local routed output rank 1 | Attention/CCL/router logically stable |
| Frozen expert original + retained, logical input | Both pass 128; first outputs equal saved reference | Logical values alone do not reproduce |
| All expert internals + physical views `bfp8_boundary_v5_experts` | Pass 128 | Instrumentation changes relevant context |
| Original class, only gate K88 `bfp8_output_k88` | Fails step 47, 602 values/rank, max 0.03125 | Spill removal alone is insufficient |
| Original class, 40 ms delay `bfp8_output_delay40` | Fails step 13, 1017 values/rank, max 0.125 | Extra host delay alone is insufficient |
| Raw GU-only retention `bfp8_boundary_retain_gu` | Pass 128 | Additional GU lifetime affects reproduction |
| Hidden-only retention `bfp8_boundary_retain_hidden` | Fails step 0; hidden rank 1 first32 columns in all8 slots | First producer is at/before hidden; physical expert input exact |
| Gate/up slices + hidden `bfp8_boundary_retain_slices_v2` | Fails step 2; gate rank 1 changes173, max0.1875; up exact | Narrows to gate/up producer or gate slice |
| Frozen original with exact physical input `frozen_experts_physical_output` | Pass128; upload bits verified | Producer padding alone does not reproduce |

All logical failing outputs above are finite; final replicas agree within each
replay. The exact physical-input frozen output differs from the saved reference
only on rank1 (2450 values, max0.0078125); comparison to the saved actual member
has not yet been measured. Source interpretation maps gate first32 columns to
sparse worker0=(0,0), also the A multicast sender. See
[AUTODEBUG_indexed_expert.md](AUTODEBUG_indexed_expert.md). No FP32 sparse
accumulator trial was attempted because a separate source allocation hazard
makes that an unsuitable control.

Commands use this module with the indicated flags:

```bash
HF_HUB_OFFLINE=1 timeout 180 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.diagnose_attention_ccl_boundaries --expert-boundaries --expert-retain gate up hidden --read-padding --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/bfp8_boundary_retain_slices_v2.json
HF_HUB_OFFLINE=1 timeout 180 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_indexed_expert_replay --output-only --physical-input --read-padding --fixture models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/bfp8_boundary_retain_hidden.pt --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/frozen_experts_physical_output.json
```

The original-class diagnostic uses `--output-only`; independent controls add
only `--gate-k-block 88` or `--replay-delay-ms 40`. All JSON artifacts retain
full commands and source hashes. Failed runs were followed by serialized
reset/list/smoke; recovery sets4 through9 all exit0.

## Isolated hypotheses

| Hypothesis | Experiment | Outcome |
| --- | --- | --- |
| BFP8 native RS/AG unsupported at H2816 | Full-attention real4096/128 runs and isolated native probes | Refuted as a blanket claim: actual execution and exact replay pass |
| Runtime FP32→BFP8 cast intrinsically unstable at the attention shape | Native cast→RS→AG,128 distinct synthetic input seeds,3 duplicate replays per seed | No differing outputs on any of four devices |
| Frozen BFP8 native RS→AG intrinsically unstable at this shape | Precast BFP8 inputs with the same128-seed/3-repeat test | No differing outputs |
| BF16 control unstable under the same harness | Both cast and precast BF16 controls | Both pass |
| Original sliding layer failure was a one-off run artifact | Repeat the original unmodified4096/128 command after healthy control runs | Same exact replay assertion fails again |

Control artifacts are `ccl_replay_bf16_cast.json`,
`ccl_replay_bf16_precast.json`, `ccl_replay_bfp8_cast.json`, and
`ccl_replay_bfp8_precast.json`, with matching `.log` files. All four processes
closed normally,exit0. Probe source:
`tests/probe_attention_ccl_replay.py`, SHA256
`fb9453754b6bbb29d4f698a0380778fac0c1d4d6f5cbd36230079f395f2620b5`.

Each local input is logical[1,1,1,2816],physical[1,1,32,2816], with independent
rank values. Inputs are refreshed between seeds before trace execution. The
trace contains native Linear RS followed by AG using the same model CCL
helper; default includes the cast, and `--precast` moves conversion to setup.
Controls cover128 seeds with an initial replay plus3 duplicate replays each.
They establish deterministic synthetic controls, not correctness of all
possible real WO values or the complete model sequence.

Example command (all four differ only by dtype and optional `--precast`):

```bash
timeout 180 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_attention_ccl_replay --dtype bfloat8_b --precast --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/ccl_replay_bfp8_precast.json
```

## Whole-layer reproduction

Both `sliding_ccl_bfp8.log` and `sliding_ccl_bfp8_retry.log` fail at runner
line308, `AssertionError: Replay is not deterministic`, after TP1 completes.
The command is:

```bash
HF_HUB_OFFLINE=1 timeout 180 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder --layer 0 --length 4096 --steps 128 --trace --check-cache --fused-tail --hybrid-experts --optimized-shared --shared-geometry 1 --grouped-moe-reduce --qkv-fidelity LoFi --output-fidelity LoFi --attention-ccl-dtype bfloat8_b --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/sliding_ccl_bfp8_retry.json
```

The retry exited1 and closed devices normally. It did not hang; no process
kill or hang-triage capture was needed. No result JSON exists because the
assertion occurs before result serialization. The log does not identify the
failing decode index, changed-element count, or whether nonfinite values were
involved. Do not claim it failed on the first decode step or infer unavailable
PCC/timing measurements.

Both failed multichip runs were followed by serialized reset/list/mesh-smoke
recovery; see `ccl_dtype_reset1/list1/smoke1.log` and the corresponding
`reset2/list2/smoke2` files. The second recovery outcome is appended below.

## Initial source diagnosis and historical next test

The independent source audit found no concrete reduction protocol violation:
BFP8 tile pages are1088-byte aligned; two workers per direction each process
11 output tiles in8+3 groups, and the tail handling is balanced. Existing full
BFP8 layer success and native controls prevent a blanket unsupported verdict.
See the independent diagnosis report when available.

Next useful instrumentation is failure metadata first (TP, step, changed
count, finite/NaN counts), then duplicate-replay boundary checks for local
WO FP32, cast BFP8, DRAM copy, RS, AG, promoted/post-normalized values, router
output, and final output. Keep device tensor handles during capture and read
only outside the trace. Retaining handles can change allocation and timing,
so preserve the uninstrumented reproducer. A model-local contrast
WO FP32→BFP8→BF16→existing reduction can distinguish pre-quantization from
BFP8 transport/reduction only after local WO repeat stability is demonstrated.
No such instrumentation or contrast was run during this bounded handoff.

Second recovery completed: reset2,list2,and FABRIC_1D 1x4 open/close smoke2
all exited0. Four ASICs visible; no locks cleared or additional reset needed.
Hardware was explicitly released to root after that smoke closed. Subsequent
authorized boundary experiments and current ownership are recorded at the top.


Watcher initial attempt (`bfp8_output_watcher.log`) does not reach the model:
fabric ACTIVE_ETH kernel configuration is 28464 bytes, above its 26624-byte
limit. It exits 1 and closes devices; no replay JSON or memory-check verdict
exists. The next bounded retry adds only `TT_METAL_WATCHER_NOINLINE=1`, preserving
NoC checks while reducing compiled check code size. Profiler and DPRINT are
explicitly unset, polling interval is 10 seconds, and logs use the dedicated
`bfp8_watcher` directory. Recovery is serialized before retry.


### Watcher result and next controls

`bfp8_output_watcher_noinline.json/.log` passes all 128 original-class
output-only steps with `TT_METAL_WATCHER=10` and
`TT_METAL_WATCHER_NOINLINE=1`. Profiler and DPRINT are unset. No NoC checking or
Ethernet checking is disabled. The Watcher log contains eight completed dumps
and no detected fault; all devices detach and process exit is 0. Runtime is
`8b59370c...`; diagnostic source is
`864d2110a16bb3a2591f3fc19aaa896161ed152e48c4193883aa0ad5d46bd74c`.
This is a scoped passing debug-kernel control, not proof the normal-kernel race
is absent. Hardware was released during subsequent CPU preparation and then
reacquired for the N2/address controls under the coordinator's standing grant.

Source auditing confirms raw GU remains a local until original `_chunk`
returns. Retaining it changes its lifetime through later shared/tail operations
and also through the next warm/capture attention prefix because the retained
dictionary is only cleared on `_chunk` entry. Gate/up slices are independent
L1 buffers; the GU reshape is a view. Ordinary attention collective payload
buffers are DRAM, so direct payload-buffer alias to GU L1 is not established.
See the parallel lifetime audit for source details.

The diagnostic now supports `--log-addresses` (host scalar metadata only),
recording warm1/warm2/capture phases, previous retained dictionaries before
clearing, and raw/reshaped GU addresses. `--expert-retain none` stores no expert
handles. `--gate-n-tiles 2` is a separate geometry control using grid6x1,
per-core/block/subblock N2, with original K44 and down program unchanged.
Neither change has been applied to the production runtime or runner.


`bfp8_output_n2k44.json/.log/.pt` uses original output-only execution with
only the gate/up geometry changed to grid6x1/N2/K44. It fails at step7 /
position4103: 1715 finite output values differ per rank, max0.75; replicas
remain equal within each replay. This geometry is not a verified workaround.
A gate/up-slice N2 checkpoint run is queued to test whether the affected
columns follow the sender's widened 0:64 ownership.


### Sender ownership discriminator

`bfp8_boundary_n2_slices_addresses.json/.log/.pt` reproduces at step0 with
first logical change in gate, now rank2. Changed gate elements occupy exactly
columns0:64 across all8 experts: counts by 32-column tile are
`[126,128,0,0,0,0]`. Gate changes254 finite values, max0.328125; up is exact.
Final outputs change1152 values per rank, max0.1875, and replicas agree.
Expanding the sparse sender worker's ownership from one N tile to two expands
the corrupt columns correspondingly. This strongly favors the sparse producer
on sender worker0 over generic slicing/GELU or a universal CCL format issue.

The same JSON contains221 scalar address records across warm1/warm2/capture.
The capture GU and its reshape both use0x174480; gate0x171fc0, up0x1717c0,
hidden0x170fc0. Retained WO starts0x1727c0, exactly after the gate's per-bank
allocation, with no overlap established. The apparent mix-weight/shared-input
reuse at0x176c80 occurs sequentially after the expert call. These observations
do not prove there is no use-after-free; they refute a direct same-base overlap
among the recorded current tensors. N2 physical expert-input padding is not
bit-stable (1822 changed words, finite max delta0); the earlier N1 hidden-only
run separately established a logical error with bit-stable physical input.

Reset11/list11/smoke11 and reset12/list12/smoke12 all exit0. The next control
is original N1/K44 with only input L1→DRAM before `_chunk`. Source confirms
`to_memory_config` uses the default tiled same-dtype copy: all88 BF16 pages of
2048 bytes are transferred without unpack/repack, preserving unused rows and
NaN bits. Extra program timing and addresses still change, so a pass would
remain a placement control, not proof of a source fix.


The lifetime investigator's CPU bit audit identifies the N2 physical input
changes precisely:1596 zero→positive-infinity and226 positive-infinity→zero
words, entirely unused rows1,2,4,5,6,7,16,17,18,19,21,22,23. The finite-pair
max-difference field of0 does not imply these words are numerically unchanged.
The logical row is exact; all ranks have the same padding transitions.

`bfp8_output_expert_input_dram.json/.log/.pt` fails at step26 / position4122:
1241 finite output values differ per rank, max0.125; within-replay replicas
remain equal. Moving only the expert input to DRAM is therefore not a fix.
The original N1/K44 gate/up program, weights and activation dtype are unchanged.
A sliced-boundary DRAM-input control is next, while source work checks whether
the preceding one-core generalized router leaves state on sparse sender core0.


`bfp8_boundary_input_dram_slices.json/.log/.pt` also fails at gate, step6 /
position4102, now rank0. All126 changed gate values lie in columns0:32
(max0.1015625); up is unchanged. Hidden changes123 values; final output changes
1302 values per replica (max0.1875), all finite. Input placement therefore
does not remove the worker0 gate pattern. This demotes an explanation confined
to interleaved L1 reads.

The next native-supported control moves the generalized router's one-core grid
from(0,0) to(1,0), along with bias, index, output and output-index buffers during
setup. The native op takes its worker grid from the input shard and validates
other tensors against it; there is no origin-core requirement. The original
sparse N1/K44/L1-input configuration is restored for this control. This tests
preceding core state separately from the sparse multicast sender's fixed core0
role. No production change is retained.


### Router placement contrast and stress

`bfp8_output_router_core1.json/.log` moves only the native generalized gate's
shard grid and four persistent buffers to(1,0). It passes128 original-class
steps with original N1/K44/L1 expert input. Restoring core0 immediately afterward
(`bfp8_output_core0_postcontrol.json/.log/.pt`) fails at step62 / position4158:
1727 finite values per replica change, max0.125. The negative control confirms
the prior failure is still present. Reset15/list15/smoke15 all exit0.

`bfp8_boundary_router_core1_slices.json/.log` passes128 steps with gate/up/hidden,
router IDs/values and physical input checkpoints plus scalar address records.
No physical-only variation is observed. `bfp8_output_router_core1_stress.json/.log`
then passes exactly1024 duplicate comparisons, eight per position across128
distinct input/position updates, with the original decoder/expert methods.
All passing processes close normally. This justifies a model-local placement
candidate, not a proven low-level reset defect or a completed accuracy gate.

The prepared frozen-prefix control allocates both core0 and core1 gate inputs
and persistent output buffers in identical order in both processes, uploads
exact saved physical expert input/routes/IDs, and changes only which native
gate executes immediately before original expert computation. The expert still
consumes frozen routes/IDs; the prefixed gate cannot change its inputs. Results
will also compare the first output to both members of the original failure pair.


### Final investigator handoff

`frozen_physical_native_gate0.json/.log/.reference.pt` and corresponding
`native_gate1` artifacts both pass128 duplicate replays. They allocate both
prefix placements in identical order; recorded prefix address maps are equal
across processes. The first logical and full physical outputs are bit-identical
between the two controls and exactly match the saved whole-layer `actual`
member. The saved `reference` member differs on rank1. This identifies which
replay value the isolated chain produces; it is not an independent arithmetic
oracle. Native gate alone is insufficient to reproduce the whole-layer failure,
so no simple persistent-reset source defect is established.

Hardware is explicitly released to the coordinator after both processes exit0
and close all devices. No production runtime, original runner or C++ source
was edited by this investigator. All experiment JSONs record runtime
`8b59370c...` for the current-policy phase. The separate diagnostic scripts are
Python-compiled and Black-checked; no C++ build is required. Exact replay/replica
checks were never relaxed. Physical-only padding variation is recorded but
does not replace the logical output contract.

The coordinator owns the remaining acceptance work: validate the integrated
model-local router-placement candidate against original numerical/reference
accuracy and all replay gates, and run full-attention/stack/batch/cache controls
and paired latency measurements. No performance or broad-correctness claim is
made from the instrumented or frozen runs. The tested placement is(1,0), with
all generalized-gate buffers on that same valid one-core grid.


Coordinator integration checkpoint: the production runtime is now
`070613ddc32b5cc8a22fd92a64cb541de9a9f27152852f1caad0843d1d06903d`. The coordinator moves only generalized-router memory and its four
persistent buffers to(1,0), immediately after router construction. Baseline
and original runner are unchanged. Ordinary4096/128 accuracy, cache, replay and
paired performance validation is underway. This is a model-local workaround
pending those gates, not a claimed source-level root-cause repair. All preceding
diagnostic observations retain their recorded old runtime8b59370c provenance;
the changed constructor allocation order requires fresh ordinary-run evidence.


Coordinator's first ordinary integrated run now passes:
`sliding_router1_ccl_bfp8.json` at runtime070613dd, real4096/128, has minimum
output PCC0.998745334, cache PCC0.999996658, exact repeated-trace equality and
replica equality for128 steps. Host TP1/TP4 decode medians are825.761/737.703us
and prefill medians221490.578/93316.212us. These are paired host measurements
from the ordinary runner, not a causal speed claim against the old failed
BF8 path. Full attention, stack, batch, maximum-context and Watcher acceptance
remain with the coordinator.

The diagnostic now defaults `--router-core-x` to unspecified (use current
production placement); explicit `--router-core-x 0` restores the old placement
for future negative controls, and `--router-core-x 1` selects the workaround.
This avoids silently treating the newly integrated core1 default as a core0
control. Prior artifact commands/source hashes retain their original meaning.
