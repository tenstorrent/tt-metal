# Optimized multichip decoder work log

Stage 5, ornith-ai/Ornith-1.5-9B revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53.
Starting clean commit b8d8d43d; stage 4 implementation f6b1e5fa. Target: 1x4
P300c Blackhole IDs 0–3, TP4, FABRIC_1D / Linear, two links. No full-model or
vLLM work. No pushes. Plugin inventory confirms enabled tt-model-bringup 0.1.23
and tt-autodebug 0.1.8. Shell setup: source /home/ttuser/ornith-9b/lane-env.sh,
which invokes the installed environment.py. No applicable AGENTS.md found.
Device ownership check: fuser found no users on IDs 0–3 before stage work.

Acceptance remains PCC >=0.995 including residual-subtracted contributions,
both layer kinds, tracked changing-input trace replay, logical unaligned lengths,
full context 262144, larger-batch stack, stress, fallback audit and watcher10.
This is a dense model: MoE requirements are inapplicable.

## Operation-topology audit before candidate measurement

Source: baseline/multichip_decoder.py.txt; prior measured tables:
../multichip_decoder/tracy/{0,3}/{prefill,decode}_perf_report.{txt,csv}.
Prior decode category sums: matmul 130.8/124.0 us, CCL 27.5/27.8 us,
movement 55.3/52.0 us, other 83.5/73.0 us (linear/full).
Prefill CCL 660.5/663.5 us is material relative to 2.16/2.22 ms whole layer.

| Boundary | Current topology | Candidate / dtype constraint | Action and evidence |
|---|---|---|---|
| Attention input | Packed QKV+gate (linear also beta/decay); BFP8/HiFi2 DRAM-sharded decode, then interleaved packed output | Compare split components, LoFi, attention-only BFP4; reduce output movement | Pending stage measurement; stage4 geometry table is control |
| MLP input | Separate gate/up decode consume identical normalized tensor; BFP8/LoFi, fused SiLU multiply | Packed DRAM-sharded gate/up, legal output splitting, larger block/readers; prefill already packed fused SwiGLU | Pending |
| Attention/MLP outputs | Row-parallel matmul -> persistent direct AR decode; prefill MM -> RS -> AG | Chunked direct prefill AR, persistent buffers; BFP8 CCL; fused MM+RS with compatible residual | Pending |
| Residual and next norm | Replicated BF16 stream; local decode L1 width shards; no layer-boundary CCL | Hidden-sharded BF16/BFP8 stream -> distributed norm -> delayed gather or fused gather+MM; preserve shard across residual add | Pending coherent family tests, no immediate restore for rejection |
| Local activation layouts | Norm 32 cores, gate/up 64, row-output 64; exact-K input shards | Carry alternative residual grid through norms/adds; phase-specific grids and precision-locked role sweep | Pending |
| Fused CCL | Prior AG+MM and MM+RS tested; fused-prefill resident NoC watcher defect, safe 64 KiB plan | Retry safe constrained fused path with compatible dtype/layout/two-link settings | Pending; prior watcher_triage/AUTOFIX.md retained |
| CCL workspace | Two shared persistent 1 MiB receive buffers/chip; two semaphores | Dynamic vs persistent API outputs where legal, prefill buffers sized by chunk | Pending; account all persistent changes |
| KV/recurrent | BFP8 full-attention KV, FP32 recurrent; logical-length aware chunks | KV BFP4 candidate, preserve page map and max context | Pending |

## Coherent topology families to compare

All families run TP4 with sharded weights/local heads; replication below means
residual replication, never replicated whole-decoder execution.

| Family | Residual input/output | Row boundary / next consumer | Payload per BF16 boundary | Packing / buffers |
|---|---|---|---|---|
| Control | replicated | MM + async AR decode; MM + RS + AG prefill; local RMSNorm | full hidden 8192 bytes/token | packed attention, split decode MLP; persistent AR |
| Chunked AR | replicated | MM + small-row AR chunks; local RMSNorm | same bytes, changed chunk scheduling | same projections; setup-owned chunk workspace |
| Hidden shards | local1024 | MM + RS; add; distributed norm; delayed AG | RS output2048 bytes/token + norm stats | packed/split with matching dtype; reuse supported CCL buffers |
| Fused hidden shards | local1024 | fused MM+RS; add; distributed norm; fused AG+MM | same sharded boundary, overlapping movement | constrained safe fused plan; packed attention/MLP |
| Reduced communication dtype | either compatible family | BF16 residual maintained; reduced normalized activation or CCL payload | BFP8 about half BF16, tile exponent overhead | compare with same topology/packing/buffer plan |

No rejection is earned by a first API error. Adapt and retry, or AutoDebug/AutoFix.

## Stop rule and timing

User-selected stop rule is >=70% roofline plus last two kept gains <3%, or
8 hours of device work from the first candidate measurement. Record exact timing
in the run ledger. Quality gates remain mandatory. At initial planning no stop condition had been met.

## Initial measurements and precision result

Fresh unchanged TP4 baseline S1024: linear prefill2.08996 ms, decode0.307854 ms;
full prefill2.18970 ms, decode0.288252 ms. Files baseline/perf_{0,3}.json.
The attention-LoFi candidate fails evolving linear decode step2: output PCC
0.994722, contribution0.994015. Gate remains0.995. This is real model weights;
no precision policy is rejected on synthetic weights. Need isolate input versus
output projection fidelity and record timing before a failing accuracy step.

Reset policy follows the explicitly selected multichip skill's failed-run rule.
No foreign device processes were present; no process was killed. During reset,
the next queue was incorrectly launched before the previous shell completed.
Stopped the queue between jobs with STOP and suspended only the recovery-driver
shell (not its running reset) so it could not launch listing/smoke concurrently.
The input_l1 attempt overlapped reset and is infrastructure-only evidence; it
will not be ranked. Await bounded jobs, finish serialized recovery, then repeat.
All later device jobs will share a filesystem flock to enforce serialization.

Source inspection confirmed that the generic DRAM-reader restriction1..3 does
not apply to this checkout's opted-in multicore pipeline: matmul_device_operation.cpp
branches on use_multicore_pipeline before that check. Prior 4/8-reader results
are therefore not evidence of a stale binary. Sweep both relevant configurations.

## Grouped projection candidate

Batched real-weight layer0→3,32 independent rows, ragged31/32/33 prefill and
17/18 continuation, shuffled page maps, four changing-input traced steps passes:
`batched_candidate_32.json`, minimum PCC0.9997884. Median grouped decode9.06157ms
versus stage4 final18.63515ms (same harness timing, historical baseline; fresh
baseline planned). The candidate batches norms/projections/residual/CCLs while
running independent attention state per row. Packed L1 residual crosses layer
boundaries without any CCL/reshard; packing/unpacking is once at stack entry/exit.
Input-attention local L1 also measured alone:0.30477/0.2856ms single-row decode,
versus0.30785/0.28825 baseline. Production implementation is being refactored to
explicit decode helpers and optional precomputed projection arguments, avoiding
the candidate's mutable Python dispatch flags. No final-result claim yet.

Prefill direct AR128 and224 pass but are slower than RS+AG. AR256 failed L1
scratch allocation (worker CB end1161920 vs first buffer1100928); retry releasing
the consumed input-normalization tensor also failed. AR224 is the successful
shape adaptation, with fewer logical rows per collective. AR64 is another
measured control. No full1024-row direct AR allocation was attempted.

AutoDebug layout investigation launched inspection-only via installed launcher.
Child sandbox preflight failed because unprivileged namespaces are disabled;
retried with AUTODEBUG_SKIP_CHILD_SANDBOX=1 under this session's unrestricted
execution authorization. Both launcher logs retained. Await report and verify
against source before repair. Device jobs remain serialized with a common flock.

## Coherent families and layout repair, 21:32 UTC

Hidden residual families now pass both kinds after adapting fused-gather workers
from eight to four per link and making BF8-input matmul outputs explicitly BF16.
Separate hidden-sharded decode is about0.468/0.449ms; persistent buffers reduce
it to0.460/0.441ms. BF8 hidden families improve1024 prefill to1.797/1.916ms with
persistent outputs, but decode0.467/0.434ms remains slower than replicated direct
AR. Fused gather/scatter BF16 pass; full tables are retained in family_* JSON/logs.
BF8 fused gather produces nonfinite values; the upcoming mixed family keeps the
fused gather input BF16 and reduces other CCL payloads toBF8. This is not a rejection
of fused CCL on its first API error. Family residuals remain hidden-sharded through
distributed normalization; they do not immediately restore old replicated residuals.

Fresh stage4 TP4 stackB32 is18.62204ms. Grouped projections/control is9.06102ms,
and truthful producer layouts plus removed redundant reshards is9.05500ms.
B1 grouped/control0.56903ms improves to0.55806ms with movement repair. All those
rows have minimum PCC>=0.9997884. Initial fresh stage4B1 reached validation and
benchmark but failed only while writing MRO provenance outside the model directory;
fixed the metadata path to repo-relative and queued a fresh run. No result is claimed
from the failed report. The production path still has serial attention per request;
fused recurrent and full-attention prototypes are not yet final defaults.

AutoDebug2 identifies the batch-attention row1 failure as singleton CoreRange
ordering: SDPA reads user4's query for user1 while retaining user1's cache/position.
Source verification confirms range-by-range enumeration. Applied rectangle-prefix
grids only; independent source check and device A/B are pending. Page-table stride
was investigated and is not the leading diagnosis. No cache dtype/context reduction.

## Batched attention, pooled KV, and hardware sweeps

Rectangle-prefix repair passes batch32:3.30521ms with minimum PCC0.9998240.
The16-step state checker initially concatenated local Q/K/V groups in chip order
rather than reconstructing global Q,K,V order. Corrected to match the existing
per-layer probe; no numerical gate changed. Private and pooled KV variants now
pass outputs, convolution, recurrent state, and sampled KV pages at batches
2,3,7,8,12,13,16,31,32. Batch32 private3.3071ms versus pooled2.5739ms,
minimum PCC0.99949396. These remain prototypes until integration/final validation.

The role-isolated LoFi passes do not measurably improve whole-layer latency;
retain HiFi2 attention. Split packed-input projections are slower:linear0.36624ms
and full0.30872ms versus baseline0.30785/0.28825ms. Column-output families pass
with hidden residuals through distributed norm but are slower. Mixed fused CCL
(BF16 gather input, BF8 reduction payload) passes; whole-layer decode is about
0.95–1.52ms and does not beat the direct replicated-residual family.

Prefill-only normalized BF8 activations plus BF8 CCL pass both kinds, measuring
1.9957/2.1328ms prefill and0.3079/0.2881ms decode. Activation-only control is
2.1296/2.1985ms. This modest candidate has not been selected or long-context signed off.

New DRAM geometry sweep adapts bank-shard padding to reader count:original linear
packed weight has13 tiles/bank, invalid for ordinary two-reader equal partition;
rounding to14 makes those trials runnable. Read counts1,2,3 and opted-in pipeline
4/8 remain separate controls. Large K block32 exceeds physical L1 at inc8/out16;
smaller blocks pass, and strictly larger blocks at identical geometry are marked
unmeasured under that measured lower bound. Other widths/geometries are still tried.

Prefill sweep initially retained its unused L1 reference input in DRAM-input
trials. This caused a misleading L1-clash failure. Revised runner holds only the
requested placement; previous DRAM rows retain allocation_policy and are not
ranked against revised coherent rows. It retries those shapes and keeps the failure.

Movement profiler collection and advice-enabled tt-perf-report complete for both
kinds (tracy_movement). Decode device-op sums288.4/264.3us; prefill2005.7/2102.4us.
Reports still advise explicit program config on four MinimalMatmul rows; these
already have config=MinimalMatmulConfig. The live sweep tests actual M/K/N blocks,
subblocks,1D/2D and placement rather than dismissing the advice. Host benchmark
loops are intentionally suppressed in --profile mode to avoid profiler-buffer
exhaustion; signposted synchronized host-window durations are recorded separately.

### Pooled vector integration and explicit head waves

The API-level wave hint did not constrain the fused chunk primitive's total work. Adapted calls now use explicit `num_heads` and `head_offset`, sharing output scratch and recurrent backing storage across disjoint waves. Independent source verification checked batch/head indexing and aliased initial/final state. Batch-32, 16 evolving steps with convolution, recurrence, and KV comparisons: 24-head automatic-producer waves 1.948594 ms; explicit 32-head single-producer waves 1.833264 ms; explicit 48-head single-producer waves 1.797740 ms. All retain the original PCC threshold. Artifacts: `chunk_pool_wave0_32.json`, `chunk_pool_explicit_wave{32,48}_32.json`, paired logs and command snapshots. The 48-head variant is selected for integration, pending final default-path gates.

Production adds setup-owned batch state pools and batched attention. Private state allocation remains supported. Pool metadata is host scheduling metadata excluded from tensor snapshots. Linear partial/reordered pools take the per-request TTNN attention path with grouped dense projections; full attention deduplicates shared backing pools and applies each request's page offset on device. The layer-to-layer packed residual stays L1 with no boundary gather/reshard/all-reduce. `integrated_jobs.json` checks default classes at batches 2,3,7,8,12,13,16,31,32, including state and trace evolution.

### Integrated state/trace gates and SDPA page-tail repair

Default pooled path passes B2,3,7,8,12,13,16,31,32 with16 evolving steps and state checks (`integrated_*state_stack*`). Default B32 before the new page-tail repair reproduces1.786201ms. B1 tracking/benchmark is0.558251ms. Both kinds pass128-step B1 stress, all logical lengths1,2,3,4,31,32,33,1023,1024,1025,2051, tracked refresh/positive-control checks, and262139/262143/262144 context gates (`integrated_gates_jobs.json`). Watcher10/NOINLINE passes B32 and both individual kinds before the page-tail repair; final-source watcher still required.

The additional mixed-pool test exposed a setup-only test bug: its post-capture allocation churn first compiled a FILL program with an immutable cache constant. Warm the exact churn/reset helpers before capture; do not acknowledge constants or disable tracking. Linear full/subset/reversed/mixed pools, bitwise neighbor isolation, interleaved traces and per-request reset then pass (`pool_contracts_retry_0.json`).

Small-context full attention failed first decode, and B32 after4095/4096/4097 prefixes failed at decode0 request8. Diagnostic outputs were identical across all four ranks but reached4.8e34; PCC's float32 intermediate overflowed. Thus the rank-agreement assertion was a downstream symptom, not unequal CCL replicas. SDPA rounds a256-token key chunk to eight32-token pages and reads masked future pages without page-table bounds checks. Existing unpadded widths3/5/4 and133 were unsafe. The decoder now pads a separate attention table to eight pages, initializes tails to a valid local page before adding pool offsets, and retains a separate unpadded cache-update table. Public logical lengths and context capacity are unchanged. Source evidence: sdpa_decode/device/kernels/rt_args_common.hpp:62 and dataflow/dataflow_common.hpp:755,837.

After repair, full-attention mixed-pool/heterogeneous-context/isolation/interleaved-trace checks pass (`pool_contracts_padded_3.json`), as does B32 4K-prefix128-step/state stress (`integrated_long_stack_padded_32.json`). The frozen stage4 long stack also passed, so the expanded pooled allocation exposed latent unsafe page-table tails. External AutoDebug report is still pending; it is being checked against these discriminating results. No gate was weakened.

The decode-mask candidate replaces tiny g/beta padding with setup masks but is slower:0.300383ms versus same-run control0.297519ms (`decode_mask{,_control}.json`); reject. Short-prefill MLP standard kernels show isolated savings but whole-layer results are shape-dependent, so no production switch yet. Batch folded-channel fused convolution initially failed its chunk-divisibility contract; retry pads dummy channels to a whole channel chunk and owns that padding.

Batch32 Tracy first overflowed profiler buffers during many setup/prefill calls and its report failed. This artifact is invalid for ranking. The retry drains device profiler buffers between prefill requests and outside captured/timed decode windows; it retains exact signposts.

### Batched page-table integration and profiler recovery

The attention-only padded table fix introduced per-request pad/add dispatch. A complete ordered pool now concatenates local tables first, pads the batch once, and broadcasts immutable setup-owned int32 page offsets. Heterogeneous/subset/mixed pools retain the general path. Candidate `table_offsets_32.json` measured1.654017ms; final default `table_default_state_stack_32.json` reproduces1.653349ms (source d3222ee33a79). Default B2,3,7,8,12,13,16,31,32 passes16 evolving state-checked steps. Both pool-contract suites pass with program-cache allocation tracking, and B32 after4Kprefix passes128 steps under tracking (`table_default_long_32.json`).

Folded-channel convolution padding repairs make128/256/512 channel chunks correct, but their2.1378/2.1411/2.1534ms stack results lose to the contemporary1.9973ms control. Reject these shapes. The completed external AutoDebug report and local fix account are in `long_batch_autofix/{AUTODEBUG,AUTOFIX}.md`; they support the SDPA tail diagnosis.

Profiler retry without MID_RUN_DUMP still dropped records. Enabling MID_RUN_DUMP removed buffer overflow, but Tracy exceeded32K source locations and omitted the decode signposts. Both collections are invalid for timing. The next attempt additionally sets TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1 to retain device CSVs without exporting every device zone to Tracy. Its completeness must be checked before use.

### Complete batch-32 profile and remaining costs

`tracy_table/stack32` is complete:254 merged operations in the signposted warmed trace,1399.448us device time +302.568us operation gaps versus1759.930us synchronized host time (57.914us remainder). CSV, text advice table and SHA provenance exist. The advice column is empty; manual topology analysis identifies six chunk-GDN calls255.154us, entry/exit format conversions, and dense projections. The list-contract harness includes one input packing and one output unpacking for the whole two-layer stack. A separate packed-I/O harness preserves the identical random inputs/PCC/state checks, refreshes the packed input outside capture, and reports that public contract explicitly. Frozen stage4 adapts to that same contract inside its timed trace; both result families remain recorded.

GDN48heads/1producer/1receiver per head reproduces1.653195ms.52heads fits the row-major fallback and reduces six calls to five:1.627637ms, all16-step state gates pass.32heads/2receivers is slower1.687303ms. Shared producer pools with fewer producers than heads fail geometry validation: source requires at least one home producer per head. Retry a legal48heads/48producer pool and compare52/55head geometries and handoff depth before rejecting the family.

### Packed API, local fusions, and warmed prefill selection

Same-contract packed I/O comparison (`packed_io_jobs.json`): B1 frozen stage4/current default0.578763/0.560818ms; B32 18.990974/1.116416ms. These include all decoder kernels and CCLs, with input refresh and host output extraction outside the trace. Both use identical prefix/input/state checks. The list API remains separately reported at18.62204/1.653349ms and includes packing/unpacking once per stack.

Valid producer-pool retry48heads/48producers/depth2 passes at1.653984ms. Source requires at least one home producer per head, so fewer shared producers cannot increase head capacity. Explicit52head per-head geometry at depths1/3/4 measures1.626140/1.628088/1.628304ms.55heads at depths1/2 measures1.637604/1.637771ms. All pass16 evolving steps and state gates. Select52/depth1 for cumulative testing.

The batch profile text (unlike the empty CSV Advice cells) advises tracing for19us of large operation gaps. The window is already traced. Reducing GDN launches6→5 and sweeping handoff geometry are the targeted response; final profiling must check the remaining gap.

FP32 convolution addcmul replaces three multiply+add pairs without reducing precision: packed B32 1.098897ms versus1.116416ms. Owned neutral GDN tails can be folded into the preceding data-moving reshapes with explicit pad_value=0.0, followed by a metadata shape exposing initialized rows. First tuple-view API rejected logical-volume expansion; the experimental Shape overload was not bound. Public reshape(logical_shape,padded_shape) adapts the API and passes. Padding-only1.097733ms; combined convolution+padding1.082689ms, all PCC/state checks unchanged. `cumulative_jobs.json` crosses these with52head/depth1 and tests boundary batches plus4Kprefix128step tracked replay before production integration.

Short-prefill candidate timings initially varied enough that five samples were inconclusive. Added explicit benchmark warmup/repeat controls, preserving old defaults. `prefill_warmed_jobs.json` measures3 warmups and25 samples per logical length, including32/33/64/127/128/129/255/256/257/1023/1024/1025/2051, with stage4/current/input-kernel/combined-short-MLP/BFP8 comparisons. The combined short kernels improve32/33/64 but input-only does not consistently improve whole-layer latency. BFP8 prefill activation+CCL passes short/boundary PCC (minimum0.999382) and improves the1024-token region modestly; max-context and128-step tests are queued before any selection. Decode precision remains unchanged.

An isolated AutoDebug source investigation is checking whether setup-reordered decode-only projection weights and the existing fused QKV-head creation operation can remove the full-attention slice/reshape sequence. Parent owns device experiments; child does no hardware work or source edits.

### Cumulative tracking and long precision gates

Cumulative B6 initially passed all16 decode/state comparisons but allocation tracking rejected a subsequent benchmark replay: post-capture KV inspection slices first compiled12 immutable program-cache metadata buffers in trace scratch. Both stack harnesses now warm those exact inspection slices before capture. Tracking/program-cache checks remain enabled and no extra buffers are acknowledged. B6 retry passes; B2,3,6,7,8,12,13,16,31,32 and4Kprefix128step combined candidates pass. `cumulative_packed_32.json` reports1.084251ms with tracking enabled; untracked crossed-wave reruns check whether the earlier wave gain survives the fused graph.

BFP8 prefill activation+CCL current-production candidate passes262139/262143/262144 context and terminal trace for both kinds, with minimum output/contribution PCC0.999241 linear and0.999602 full (`precision_long_{0,3}.json`), plus128-token prefix/128-step stress (`precision_probe_{0,3}.json`). This is candidate evidence, not yet the production default.

Warmed MLP-only comparisons support a phase-specific short policy: separate standard1D gate/up at<=64 rows and2D at65..128, preserving minimal fused SwiGLU for larger chunks. At32/33/64 rows, linear current0.8471/0.9482/0.8799ms versus MLP-only0.8269/0.8813/0.8332; full0.8767/0.9383/0.9286 versus0.8332/0.8871/0.8762. Input-projection replacement offers no consistent additional whole-layer gain; do not integrate it.129..256 gains are small/inconsistent, so retain the existing larger-chunk kernel family.

### Recorded-activation precision decisions and QKV gates

Precision candidates now use real model weights and recorded HF teacher-forced activations from the first four decoder layers, for two original text corpora, lengths32/33/128/1024/1025, and16 continuation steps. This is decoder calibration, not full-model generation or an accuracy claim. The CPU-only recording script, exact text, model revision and tensor SHA are in `recorded_precision_input_provenance.json`; inputs are reproducible. Both output and residual-subtracted contribution PCC must remain>=0.995 on every rank. These results supersede rejection decisions based solely on synthetic activation stress, while existing stress gates remain unchanged.

|Policy|Layer|Pass|Minimum output/contribution PCC|Median traced decode ms|Failed comparisons|
|---|---:|---|---:|---:|---:|
|control|0|True|0.999965|0.297572|0|
|control|3|True|0.999119|0.267587|0|
|down_bfp4|0|True|0.999741|0.293750|0|
|down_bfp4|3|True|0.997034|0.263848|0|
|gate_up_bfp4|0|True|0.999890|0.280211|0|
|gate_up_bfp4|3|True|0.997146|0.249954|0|
|input_bfp4|0|True|0.999750|0.286897|0|
|input_bfp4|3|False|0.994235|0.261477|7|
|input_lofi|0|True|0.999953|0.296946|0|
|input_lofi|3|True|0.998422|0.267142|0|
|kv_bfp4|3|False|0.973400|0.267554|90|
|output_bfp4|0|True|0.999466|0.295601|0|
|output_bfp4|3|True|0.997373|0.265749|0|
|output_lofi|0|True|0.999972|0.297491|0|
|output_lofi|3|True|0.999427|0.267621|0|
|phase|0|True|0.999922|0.297547|0|
|phase|3|True|0.999119|0.267729|0|
|phase_attention|0|True|0.999966|0.297543|0|
|phase_attention|3|True|0.999119|0.267670|0|
|phase_ccl|0|True|0.999916|0.297575|0|
|phase_ccl|3|True|0.999119|0.267644|0|
|phase_mlp|0|True|0.999966|0.297536|0|
|phase_mlp|3|True|0.999119|0.267576|0|

Passing BFP4 weight candidates are hand-off evidence for stage8, not defaults in this optimize stage. BFP4 KV fails the recorded contribution gate and is rejected. Global input LoFi passes recorded inputs but fails an existing synthetic decode contribution gate; a phase-specific prefill-only LoFi candidate is being checked without changing that gate. Phase-specific BFP8 activation/CCL policy passes recorded inputs; final op dtype provenance is still required.

Native QKV-head creation uses the existing packed Q/gate/K/V weights as eight query-sized heads, then a metadata view exposes only the first four logical query heads while preserving the padded32-head stride. Batch1 retains its original path. B2/3/6/7/8/12/13/15/16/17/31/32 pass16-step state checks under immutable program-cache tracking; the4Kprefix128-step case also passes (`qkv_gate_jobs.json`). B32 packed decode1.068014ms, versus cumulative fusion1.084251ms and earlier production1.116416ms. Direct-native projection output measured1.065776ms versus interleaved1.067426ms in isolated tests; this small difference does not yet justify its extra layout contract.

Crossing48/52 GDN head waves and depths1/2 on the fused graph gives1.0832–1.0840ms, with no reproducible52-head benefit. Keep48-head automatic depth for the simpler integrated policy.

### Packed profile, matmul cross-check, and GDN fidelity contract

The complete QKV packed-I/O profile contains118 merged ops:987.5505us device time+114.623us operation gaps versus1173.288us synchronized host window. Top families are10matmuls258.329us,5chunkGDN224.714us,4all-reduces58.774us,19slices50.417us,5FillPad44.114us,7norms44.005us,3tilizes39.131us. Producer-owned neutral padding still lowers to FillPad kernels; it is not free. Text advice suggests tracing for28us of large gaps; this is already a traced window. Further history-layout and GDN launch work targets that cost. Tables/CSV/provenance: `tracy_qkv/stack32`.

Isolated matmul winners were crossed into whole-stack candidates, including added input reshards and a separate family preserving producer layouts. None gives a reliable improvement over1.068014ms QKV control: gate-reshard1.081427,down-reshard1.072067,output-reshard1.067691,all-reshard1.084478; input-preserve1.068840,gate-preserve1.074633,down-preserve1.068394,output-preserve1.067906,all-preserve1.072666ms. Initial input-only attempt failed in candidate construction before any matmul; moved configuration to from_state_dict and queued its retry. The existing actual-core output geometry and producer layouts remain selected.

Pooled GDN fidelity probe hit an explicit numerical kernel contract, not an unsupported input shape: `chunk_gdn_compute_config.hpp:31` requiresHiFi4,FP32destination,andexactmath on every path. Production `self.compute` isHiFi4; attention projection HiFi2 is a separate policy. No assertion was removed and no lower-fidelity result is claimed. Known-invalid queued variants are stopped between jobs. A legalHiFi4 recorded-batch control will validate the new pooled real-activation harness.

Prefill input LoFi alone passes single-layer boundary/max-context gates but fails the128-step4K batch32 stack atdecode4,row12,PCC0.993812. Kind-isolated follow-ups distinguish linear recurrence sensitivity from full attention; no quality gate is weakened.

### Fusion defaults integrated; fresh validation in progress

Production SHA`da8f4bd474675698af4e162ddbc0afb9e9df17467a7a41480ec17b8913715d8f` integrates FP32addcmul, producer-owned neutral padding, native QKV head creation, standard short-prefill MLP kernels through128 logical rows, and BFP8 activation/CCL payload for chunks>=512 rows. Matmul outputs and residuals stayBF16; weight/KV precision and projection fidelities are unchanged. GDN staysHiFi4 and48-head waves. The prior production source is frozen exactly as `tests/multichip_pre_fusion_reference.py` (SHA d3222ee33a7965d56c658729bfe653d71ebe3d07d1da89f3d142c8b8151f208c), and local-fusion/phase candidates now inherit that immutable source so ongoing before/after experiments cannot silently adopt the new defaults. `fusion_default_jobs.json` reruns all boundary batches,4K128steptracking,prefilllengths,andmaxcontext on the actual new default. These results are pending at this entry.

AutoDebug identified wider SDPA grids reading Q from compute-width-based reducer coordinates instead of the actual8-wide Q shards. Batch8 is the first mismatch. The ordered sub_core_grids adaptation retains native QKV and K/V-cache update layouts and puts the32 Q cores first in compute enumeration. Source verification and128-step adapted grid/chunk comparisons are in progress; no width10/11 rejection is based on the first failure.

### Longer answers, default watcher, and long-prefill measurements

Frozen stage4 and both candidates pass1024 evolving decode steps after4Kprefix atB32, including final convolution/recurrent/KV states. Same-harness packed timings after the long run: stage4 19.857979ms, QKV1.321817ms, phase1.338092ms. Minimum PCC is0.998261/0.998450/0.995975 respectively. These are labeled candidate numbers; the combined production default must rerun this extended test before final reporting.

Current fusion default passes separate watcher10/NOINLINE runs forB32 and both representative kinds. `fusion_profile_{0,3}` complete; report generation is queued under the same serialization lock. Recorded-input controls pass for both kinds and now log actual matmul/CCL input/output dtypes and supplied kernel configs. BF8 reaches long-prefill matmul inputs and RS/AG payloads. Metadata-instrumented prefill times are diagnostic only; uninstrumented probes/stack runs rank latency.

Longer uninstrumented prefill comparison uses3 warmups+25 samples per length onTP4 and the same reference.

|Layer|Logical length|Stage4 prefill ms|Fusion default prefill ms|
|---|---:|---:|---:|
|0|4095|9.321128|9.078035|
|0|4096|8.729700|8.494862|
|0|4097|9.586796|9.332530|
|0|8191|18.286548|17.832883|
|0|8192|17.555574|16.936502|
|0|8193|18.592923|17.978961|
|3|4095|10.958744|10.668741|
|3|4096|10.425777|10.176511|
|3|4097|11.894440|11.574429|
|3|8191|25.558617|24.917353|
|3|8192|24.530364|23.649023|
|3|8193|26.308428|26.023073|

Dedicated sigmoid-gated RMSNorm/head-concatenation is numerically valid but slower: B32 1.097594ms versus1.068014ms QKV control. It requires32-token input/gate tiles and an extra multiply to implement SiLU; the added gate padding/layout work erases the fusion benefit. Reject this concrete dedicated-op family, not its first API call.

Tiled persistent convolution history first exceededL1 in cache update (5,355,456B required versus1,572,864B/core). Retry divides setup-time updates into256-channel blocks, concatenates results, and copies back into the original pooled backing. The retry is rebased onto the fusion production default and preserves canonical state reads in its dedicated validation harness. No production history change is selected yet.

### Wider-grid integration and alternative recurrent families

Ordered SDPA11x10/chunk128 passes B2/3/7/8/15/17/31/32 tracking and all128-step64K cases. At4K,8x8/chunk256 control1.271819ms versus11x10/chunk1281.258012ms. At64K the same family control is4.448004ms versus4.101303ms (chunk2564.109638,chunk5124.111798). Select11x10/chunk128 with explicit Q-first compute-core enumeration and owned4-page table alignment. Production SHA`2c5d7c9bb057dbfc0cd1faaaef99cf6fb42346eded7a08fe4d61beb6055144b5`; integrated boundary/pool/4K/64K/1024-answer checks are `sdpa_integrated_jobs.json`. This does not change the target4K roofline workload. Previous fusion source is frozen exactly in `tests/multichip_fusion_reference.py` (da8f4bd4...) so history/phased/algebra candidates preserve their original control. Batch1 attention and prefill are unchanged.

Phased GDN prep+scan is a legal HiFi4 alternative to the fused head waves. Adapted scans to <=104 heads (13 requests) instead of rejecting the110-core limit. It passes16-step pooled state checks and tracking atB32: wave13 1.378802ms, wave8 1.419324ms, wave6 1.480022ms, versus fusion default1.067789ms. Extra intermediate DRAM traffic and slicing outweigh fewer head waves. Reject this measured family.

Direct one-token FP32 recurrent algebra over all256 pooled heads also passes state checks and tracking. Batched-matmul form1.294985ms; elementwise/reduction form1.532228ms. Both include normalization, GQA expansion, scalar layout conversions, state update/copy, and output conversion. Reject both versus1.067789ms fused control; no lower-fidelity kernel assertions were bypassed.

Tiled-history chunked update passes16-step decode atB32 (1.048268ms) andB6 (.744779ms), but warmed pooled prefill exposes a new failure: S32 outputPCC.913660,contribution.904867,rankagreement.932537. Production RM pool passes the same test. Candidate remains excluded; AutoDebug/AutoFix investigation recorded in `history_audit`. A decode-only speedup is not acceptance.

All six long recorded-HF precision checks pass (4095/4096/4097 prefixes,128 decode steps, both layer kinds; `recorded_long_precision_jobs.json`). This supports the real-activation evidence but does not waive the existing synthetic stack failure for linear prefillLoFi. Attention projection fidelity remainsHiFi2 in the production path.

### Pooled B32 coherent-family recheck and prefill trace advice

Rechecked hidden-residual families on the pooled attention stack with packed I/O. The residual stays1024-wide per rank through both layers. Separate RS/distributed norm/AG is1.344417ms; persistent RS outputs1.329495ms. Fused gather2.349413ms, fused scatter2.396023ms, both3.393444ms, mixed BF16gather/BF8CCL3.393907ms. These full families remain slower than the then-default1.049989ms replicated-residual stack. Initial harness setup incorrectly required AR buffers on hidden-residual candidates; fixed it to verify shared CCL ownership and require AR buffers only on the replicated family, then reran. No initial API rejection is claimed. BF8 hidden activation/CCL fails an existing evolving stack gate atdecode4,row3,PCC.994702; both layer kinds pass the independent recorded-HF B32 tests. The failed stack gate remains unchanged.

Prefill report advice about host gaps was tested with fixed-shape caller-managed prefill traces. Both kinds pass all3fresh inputs/page tables and state/output checks at32,33,127/128/129,1023/1024/1025,4095/4096/4097,8193 under allocation tracking. S32 eager/traced:linear.838/.617ms,full.854/.593ms. S1024:2.059/2.001 and2.175/2.129ms. At4K gains are small; trace is a caller-contract hand-off, not an unreported change to default prefill API. `prefill_trace_{0,3}.json` records all rows. The current phase-dtype matmul sweep additionally follows the report's geometry advice using actual BF8 normalized inputs and BF16 outputs.

### History AutoFix discriminating results

AutoDebug (`history_audit/AUTODEBUG.md`) found no supported wrong tap/request mapping and refuted stale update_idx/padded-user overrun in the inspected updater. Independent source check agreed. Focused state write/read/reset passed; a separate-program-cache compute-config key issue was identified but its alternating-config trigger is absent here, so it is not claimed as the cause.

Controlled prefill diagnostics show the failure is not specific to TILE: a second ordinary RM decoder with separate collectives also fails from repetition0/request0. Per-request synchronization and output cloning do not repair it. Sharing the CCL workspace, as the production stack does, passes every request and all18repetitions. Corrected shared-workspace measurements pass all32/33/1024/1025 lengths. S32 private/RMpool/TILEpool20.535/25.653/36.290ms; S1024 65.201/64.994/67.956ms. This isolates the separate-instance setup confound but does not yet identify its underlying kernel; `workspace_probe` is queued to locate the first diverging operation. Production retains its shared-workspace stack contract.

TILE history's eight256-channel cache updates are now compared against direct32-byte aligned row writes into the persistent BF16 tiled pool. Initial Python shape-slicing error occurred before kernel execution and was fixed; retry passes B32/B6 state/tracking checks. Queued sentinel tests pass under watcher for both batch sizes. The direct writer never rewrites another request's row. Long warmed prefill comparison is running.

### Model-specific kernel candidates (not defaults yet)

A Q/K-only adaptation of the repository's Gemma4 fused norm/RoPE stages normalizes all256 dimensions, rotates exactly the first64, copies the192-dimension tail, and leaves V unchanged. Native sharded Q/K stay in place. B32 1.004575ms versus SDPA-default1.049989ms; B6 .702710ms. Existing raw recurrent kernels, invoked through generic_op with explicit head offsets and persistent FP32state alias, remove the public wrapper's redundant conversions/copies while retaining all model-side normalization/layout costs. Wave104 B32 .986960ms (wave110 .987135, wave64 1.005791) versus fusion-control1.067789ms. All pass16-step state/tracking checks. Their cumulative combination with tiled history/direct writes is.912633ms atB32 and.709236ms atB6. The small-batch chunk-GDN control, every batch1..32,4K128-step,1024-answer,64K,andmixed-pool checks are queued in `kernel_full_gate_jobs.json`. No final result or selection is asserted before these complete. Kernels and Python sources are retained; initial partial-RoPE kernels have a separate immutable manifest, subsequent job provenance snapshots include all local C++/headers.


### Direct-history prefill and separate-workspace isolation

Direct tiled-row writes pass shared-workspace warmed prefill at32,33,1024,1025,4095,4096,4097. AtS1024 RMpool64.987ms/TILE65.846ms; S4096 270.881/273.739ms; S4097 296.865/299.329ms. This small setup/prefill cost is retained in the comparison against decode gains; it is not omitted as an inconvenient candidate result. Full output/contribution/rank checks pass. The initial sentinel watcher test is being strengthened with host-only canonicalization to avoid using another device permutation as the oracle.

`workspace_probe_separate` matches normalization, packed input projection, GDN norm, and output projection exactly, then diverges at the first collective reduction (rankPCC about.909–.924). Shared-workspace probe matches all operations exactly. Independent source inspection finds that current RS common-argument callbacks refresh all buffer/semaphore addresses and default AG bindings/overrides refresh their caller-owned addresses. A naive frozen-address diagnosis is not supported. Further isolation holds tensors and buffers fixed while varying semaphore ownership/cache state. The optimized stack continues to use its passing shared-workspace contract.

A further model-specific candidate accepts the flat BF16 convolution result and gates directly in the recurrent launch, preparing FP32 L2-normalized Q/K, GQA head selection, decay and beta on-core. This targets the remaining normalization/head-layout/scalar-conversion operations without changing HiFi4/exact FP32 recurrence. `flat_recurrent_jobs.json` is queued; no performance or acceptance claim is made before its tests.


### Extended kernel gate failure and AutoFix

Cumulative custom kernels pass every batch1..32, both mixed/subset/reset pool contracts,4K128-step and64K128-step state checks. They fail the distinct1024-answer input trajectory at`decode_79_30`,PCC0.994103 versus unchanged0.995. The failure is captured in`kernel_extended_answer1024_32.log`; clean close and mesh recovery pass. This candidate is excluded from production despite its shorter-run gains. The two trajectories use different allocated capacities/page-table draws, so128-step success is not evidence contradicting the step79 failure in the1024-answer test.

AutoFix source analysis finds mathematically equivalent but differently rounded normalization, decay and recurrent accumulation in raw versus chunk GDN. In particular, raw computes output from packed updated state; chunk computes output from old state plus correction and accumulates the two state-update products differently. `kernel_answer_isolation_jobs.json` separates partialRoPE,raw-only(historyRM,defaultwave110),history-only,andcumulativechunk. An independent recurrence-order repair is being prepared under new files. No epsilon, precision, fidelity or PCC threshold is relaxed.

The flat recurrent helper now explicitly guards1..32requests,exactstategeometry,1..110headwave,TILE/interleaved inputs and owns its reader scratch page. A direct PyTorch recurrence test coversB1 (which the decoder dispatch otherwise bypasses),zeroQ/K,negativeg,stateupdates,andwave/faceboundaries. This test runs separately underwatcher. A further direct tiled convolution candidate fuses four FP32tap products/SiLU with history advancement, crossing chunk/raw/flat recurrence families; no default selection is made yet.


### Flat-input first results and watcher repair

Flat-input preparation passes16-step state/tracking decoder checks atB32 (.838666ms) andB6 (.637450ms), versus cumulative raw.912004/.709213ms. Its long-answer gates remain pending. The direct helper watcher test exposed a real DRAM read-alignment defect atB2: sourceoffset32mod64 versus destinationoffset0mod64 for a32-byte read. B1 passed; production decoder inputs in the first passing runs wereL1. The reader now uses aligned64-byte DRAM row-pair reads and extracts the wanted32-byte face row into its zeroed tile; theL1branch is unchanged. NoCerror/kernel/waypoint evidence was dumped bywatcher before abort and is retained in the failing log; watcher host files were copied to`watcher_flat_direct_failure`. Automatic recovery reset and mesh smoke passed. Recovery now copies watcher dumps/kernel maps before any reset/smoke, to preserve these files for future failures. `flat_recurrent_watcher_retry_jobs.json` reruns the complete direct boundary/zeroQK/state test; no watcher-clean claim is made until it passes.

Separate-workspace fixed-buffer isolation also produced useful non-cache evidence: ABA passesallRS/AGcalls; BABfailsfirstBonly atRSrank0(PCC.901058) orAGrank3(.935514), thenAandlaterBpass. Cache-disabled resultsareidentical. Inserting anRSdownload betweenRSandAG makes the real-projection separate-workspace probe pass. These findings do not support a simple stale cached address; explicit synchronization, retainedRSintermediates and persistent-output variations remain under investigation.


### Accumulation-order repair and remaining profiler costs

Raw-only fails the same1024-answer trajectory; partialRoPE-only,history-only,andcumulativechunk allpass (1.254255/1.302955/1.233028ms). Reusing chunk's one-token scan accumulation order while retaining raw external preparation repairs the failure: minimumPCC.995991,1.164026ms; shortB32.920210ms. See`recurrent_audit/AUTOFIX.md`. Flat-input plus repaired scan is a new candidate, not an assumed cumulative pass.

The cumulative raw profile (`tracy_kernel/stack32`) has109 mergedoperations,840.082us device time and104.515us gaps against1.019585ms synchronized host time. Tenmatmuls take257.657us; fourgeneric kernels149.617us; fourARs59.461us;14binary55.813us;4reshape-codegen49.487us;7norm41.118us;6transpose35.945us;2FillPad22.526us. All per-row advice fields are empty; these measured movement costs motivated the flat-preparation and direct-convolution candidates. The profiled candidate fails the long-answer gate and is explicitly not final performance evidence.

Direct tiled convolution passes16-step output/state/tracking atB32/B6. Withchunkrecurrence:.903331/.615023ms;raw:.832030/.629944ms;flat:.769466/.576215ms. Long-answer,context,pool,andwatcher checks are queued for the chunk form; raw/flat forms remain excluded until repaired accumulation is integrated and retested.

A prefill retry wrapper initially treated `--output` as an ambiguous abbreviation of its new flags. Fixed `allow_abbrev=False`; parser-fixed retries have new labels. An edit to the live shell recovery helper caused one transient EOF while it was being read; the current helper passes `bash -n`, and subsequent ordered-decoder hardware checks pass. Future shell helper edits must replace files atomically or wait until no recovery process is running. This control-script failure is retained in the queue log and does not count as an optimization rejection.


### CCL semaphore ownership repair

AutoFix localized separate-workspace corruption to semaphores initialized on the inherited fixed8x8 grid while two-link CCL worker selection uses the actual11x10 grid. Default workers include(8,0) and(10,0), outside that allocation. Initialization is already blocking and cached address callbacks refresh correctly; adding synchronization or persistent buffers did not repair the general case. The model-local CCLManager now allocates its semaphore grid from compute_with_storage_grid_size(). No shared GPT-OSS source is changed. Eight cold BAB changing-input probes cover RS/AG, cache enabled/disabled and fixed/automatic buffers; all pass after the repair. Bank1/ping-pong eight-cycle probes and the separate-workspace watcher probe also pass. See history_audit/AUTOFIX.md, ccl_grid_fix_jobs.json and ccl_grid_stress_jobs.json. Production SHA8d1f4c57229f29e23703059b21355e4da9336a77c4df7f019981d56bad907063 includes the fix; final integrated default gates remain pending.

### Ordered flat preparation and direct convolution

The flat-input helper with the original recurrence ordering fails the same long-answer gate and remains rejected. Combining its preparation with the accepted chunk scan ordering passes1024 answers:1.089332ms,minimumPCC.995408. Adding direct tiled convolution passes at1.035655ms. Removing the two redundant gate FillPad operations passes at.999074ms,minimumPCC.995609; shortB32 is.755810ms. The custom reader extracts only valid gate scalars and initializes its own inactive rows. Flat-ordered direct PyTorch output/state comparison passes under watcher, including the corrected aligned DRAM reader. These are candidates, not final default numbers. The complete1..32 batch, long-context, mixed-pool and watcher matrix is in best_kernel_gates_jobs.json.

### Output-normalization fusion: live hang and retained evidence

The first further fusion run stalled at rounded.wait_front(4). Full live tt-triage evidence and exact failing sources were preserved in postnorm_hang before any reset or termination. The existing900-second timeout is allowed to end the process; other queues are paused between jobs with STOP. Independent AutoTriage finds all416 active workers at CB20 pushed4/popped4: NCRISC produced the first four pages, then TRISC2 incorrectly reused that CB with its own first cumulative publication count4. CB21 has the same latent producer-ownership defect. The repair dedicates CB29/30/31 to compute output stages and retains NCRISC ownership of CB20/22 when reusing those inputs for gamma/gate after preparation. Unit and identity are published before these second reservations to avoid a dependency cycle. This first hang is not an optimization rejection; a repaired candidate and full correctness checks follow recovery. See postnorm_hang/AUTOTRIAGE.md.

The postnorm repair also preserves the separate binary_ng multiply path's BF16 SiLU intermediate pack. Source evidence: binary_ng_program_factory.cpp allocates MUL activation CB3 with a_data_format, and eltwise_utils.hpp packs SiLU before multiplication. CB29 is reused by its same compute producer after its prior rounded-output pages are consumed; no producer changes ownership. A direct PyTorch plus exact-TTNN postprocess probe is being added.


### Postnorm repair, one-launch head loop, and production integration

The original postnorm run expired with124 after900s; preserved live triage precedes reset. Recovery passed. Reader-owned CB20/22 and compute-owned CB29/30/31 repair the counter alias. The fused postprocessor now preserves BF16 SiLU packing as well as BF16 recurrent-output and RMSNorm packs. Independent source review also identified compute_kernel_hw_startup's documented once-per-launch requirement; repaired postnorm and head-loop kernels initialize once and use explicit format reconfiguration thereafter. No thresholds changed.

Repaired postnorm B32/B6 .725596/.545309ms;1024-answer .975480ms,minPCC.995908. One-launch head-loop alone .686539/.559491ms;1024-answer .935773ms,minPCC.995609. Their cumulative candidate .666322/.545246ms;1024-answer .916042ms,minPCC.995908. Direct postnorm watcher probe initially failed only its optional padded-view diagnostic API; explicit logical/padded metadata view repairs it. Retry passes640 rank/batch/step records with independent Torch output/state comparison and exact separate-TTNN postprocess control. Combined post-loop direct watcher also passes640records, including DRAM/L1 reads,zero/tinyQK,signedgamma/gates and nonaligned batches. Loop-only direct watcher passes its independent recurrence oracle.

The cumulative post-loop candidate is integrated into production under tt/ with model-local kernels. Current source SHA5cdc9e59dc6c69e3f43284694ad4a225ac340f0c291f42060de94c8a156958a4. The integration_audit contains dependency/ownership proofs and589dispatch plus3520head-scheduling host checks. PrivateB1 is unchanged; complete ordered pools use fused tiled convolution and ordered recurrence/postnorm, while incomplete/mixed pools use canonical per-request extraction/direct row write. Context JSON records TILE history's393216bytes/chip/layer for B2..32 versus12288*B previously, unchanged privateB1/B32 and unchanged all-layerB1 budget. No context reduction is claimed. Driver provenance now includes tt/kernels sources. Canonical stack validation understands both old RM and new TILE history and retains all existing PCC/state checks.

Default boundary matrix1..32 passes. Initial default measurements: B1 .560746ms, B6 .545082ms, B32 .666156ms;4K128-step .878055ms,minPCC.997169. Remaining default long/pool/watcher/context/profile gates are queued; these are not final-stage signoff.

### Gate preparation and larger prefill chunks

Preparing beta/decay independently inside each head's recurrent loop removes external gate operations but costs more atB32: .678173ms short/.927287ms1024-answer versus .666322/.916042 for post-loop;B6 improves .545246→.533507ms. MinlongPCC.995367 still passes. This candidate is not selected. Its independent packed-field/CB ledger audit found no ownership or ring-pointer bug; direct watcher probe is queued. A new family prepares all batch gates once inside the convolution launch to avoid repeated per-head gate work; no result is assumed.

Chunk256/512 are slower than1024. Chunk2048 control improves S4096linear8.50→7.77ms,full10.21→9.61ms. First tuned2048 attempt hit measured L1 conflicts;11x10grid adaptation was tried. The first wider-grid jobs failed at their initial511-row control because their frozen standalone constructor still used the old8x8CCL semaphore allocation. After applying the verified CCL grid fix to that experimental wrapper, control4096passes:linear6.867ms/full8.763ms atS4096, versus fresh1024controls8.576/10.211ms. Wider tuned2048passes7.387/9.221ms. Wider tuned4096hits a genuine1712064-byte static L1 requirement above1572864 capacity; smallerK-block retries are queued. New8191/8192/8193/16385references and chunk2048/4096/8192tests are queued before selecting a new public default. Public logical lengths remain unrestricted.

### 2026-11-13 02:18 UTC — integrated post-loop gates and remaining candidates

Production SHA `5cdc9e59dc6c69e3f43284694ad4a225ac340f0c291f42060de94c8a156958a4` passed all 38 jobs in `postloop_default_gates_jobs.json`: batches 1–32, 4K/128-step and 64K/128-step continuation, both pool-contract kinds, and separate batch-6/32 watcher runs. Traced warmed two-kind stack: B1 0.560746 ms, B32 short 0.666156 ms, 4K/128 0.878055 ms, 64K/128 3.726341 ms. These are integrated-default evidence, but final selection and longer-answer gates remain pending.

Head-loop geometry HiFi4 waves 80/86/96/104/110 all passed short and 1024-step answer gates. Long-answer ms: 0.952968/0.915860/0.913667/0.915911/0.916659. Wave96 is only 0.245% faster than104; keep104 pending broader evidence. HiFi2 short passed at0.662181 ms but long-answer failed; LoFi failed short and long. Preserve unchanged quality gates; exact failures and commands in corresponding `postloop_geometry_*` logs/command JSONs.

The packed per-head gate candidate's direct watcher probe passed all default-wave104 cases but failed wave7/B17/step5: global output PCC0.9999988, minimum per-head0.719433; state minimum per-head0.9999987. Failure tensors preserved. AutoFix independent source/tensor investigation assigned; this is not a clean direct-probe result and does not justify promotion.

Larger prefill chunks: both4096 control paths pass4097 logical tokens. First failing case is8191, involving a second large chunk. Source audit identifies previous-chunk `h` surviving during the next norm allocation. Minimal M4/K16/N8 CB bytes/core802816; K8 saves270336, larger than71104-byte reported collision. `prefill_chunk_audit/AUTOFIX_SOURCE.md` gives source and arithmetic. Discriminating early-release and reduced-buffer candidates follow; default remains1024 and context262144 is unchanged.

Queued additional precision experiment: BF16 persistent pooled recurrent state (CB5/7 plus matching DRAM reader/writer pages) with FP32 HiFi4 scan arithmetic, FP32 canonical private states, and explicit conversion at pool read/write. This halves persistent pooled recurrent read/write traffic but must pass unchanged state/output/long-answer checks. No production precision change.

### 2026-11-13 02:27 UTC — target profile and arithmetic repair

`postloop_profile_target_jobs.json` completed (546.8s including9min capture/postprocessing; this is not9min of device kernels). `stage5_perf_reports --collection postloop_target --stack --measurement .../postloop_profile_target32.json` generated advice-enabled text, CSV, rawCSV links and provenance. The61-operation measured region sums855.7885us device kernels and53.957us gaps; profiled host trace0.970663ms. Main rows: SDPA211.747us; fused ordered recurrence167.756us; convolution9.940us; partial norm/RoPE22.241us; ten projection matmuls257.607us; four asyncAR64.364us. All ten matmuls are reported optimized; advice CSV is empty. They actually use BF16/BFP8, attentionHiFi2/MLPLoFi, eight DRAM readers, 71.6–84.9% reported DRAM utilization. This does not establish whole-decoder70% roofline. The profile uses final validated positions4115–4118 (prefix4095 plus per-request continuation and3steps), while the1024-answer benchmark uses positions5135–5138. Provenance now records these actual positions instead of incorrectly equating position with prefix length.

The integrated default1024-answer run passes at0.915430ms, minPCC0.9959075. Both individual-kind contracts/stress and mixed-pool watcher runs also pass (`postloop_default_extra_jobs.json`).

Conv-wide gate fusion passes shortB32/B6 and1024-answer checks at0.647055/0.527657/0.895538ms, but its independent direct probe and precise-arithmetic repair remain pending. Packed per-head gate failure was reproduced in FP64, so it is not a PCC implementation artifact. Source analysis found sigmoid selecting the FP32 numerical branch instead of TTNN's BF16 branch, and FP32 intermediate unpack defaulting toTF32. Isolated repaired candidates preserve exact gate stages and unchanged0.995 comparisons; `packed_gates_audit/README.md` and `precise_gate_repair_jobs.json` retain the diagnosis, source/API proof, and upcoming commands.

The profile's167.756us recurrence motivates a dataflow-only experiment replacing thousands of scalar RISC zero stores with the repository's NoC local zero-fill API. Each zeroed window has a barrier before reads/writes/push, and FP32 arithmetic/state remain unchanged. `fast_zero_jobs.json` includes short/long decoder gates and independent watcher stress. A separate residual+norm API comparison retains the unnormalized residual needed after theMLP; its public API returns only the normalized result, so the experiment explicitly includes the extra residual add rather than claiming a removed operation.

### 2026-11-13 02:37 UTC — BF16-state format adaptation

Initial BF16 pooled-state jobs produce invalid/constant first decode outputs, not a clean precision rejection. Source inspection finds the shared scan explicitly hoists format reconfiguration under the invariant that every scan CB isFP32 (`chunk_gdn_math.hpp:1052`). Merely changing CB5/7 dtypes violates that invariant. Retry stages BF16 state from reader-ownedCB28 into compute-owned FP32CB5, keeps every scan input/intermediateFP32, and changes only the final state pack toBF16 with an explicit pack-format reconfiguration. Reader/writer DRAM pages remain2048bytes. CB28 owns16 BF16 input pages then serves as reader scratch only after compute pops them; no CB gains a second producer. Fresh commands are in `bf16_state_staged_jobs.json`. No default precision change.

### 2026-11-13 02:53 UTC — recurrence movement and prefill lifetime discriminators

Production remains SHA `5cdc9e59dc6c69e3f43284694ad4a225ac340f0c291f42060de94c8a156958a4`; these are isolated candidates, not final default numbers.

- `fast_zero_jobs.json`: replace scalar volatile clearing of reader CBs with bounded NoC zero writes and explicit completion barriers. B6 0.528333 ms; B32 0.614739 ms versus production 0.666156; 4K-prefix/1024-answer B32 0.863661 versus 0.915430 ms. Long-answer min PCC stays 0.995907545. Direct watcher passes all boundary batches with bitwise recurrent-state control (`fast_zero_direct_watcher.json`, 92.98 s). Candidate eligible for cumulative selection.
- `pooled_one_long_jobs.json`: matched B1 long-answer control 0.588612 ms, pooled recurrence 0.568779 ms; min PCC 0.9989225/0.9990641. Short pooled B1 0.540365 ms versus 0.560746, watcher passes. The B1 pool costs extra convolution history memory; not integrated before context accounting and full-attention B1 comparison.
- `lifetime_chunks_jobs.json`: release old chunk/norm temporaries, then stage large norms DRAM -> BF8 -> L1. Both layer kinds pass 8192-token chunks through logical 16385, including non-aligned boundaries. At S8192 linear/full warmed prefill 12.798/19.945 ms versus lifetime-only4096 14.312/21.573. At S16385 28.447/57.730 versus 30.072/59.283. Selective staging above4096 and max-context retests queued. Further16K/32K chunks retain large activations in DRAM and have fresh-reference boundary jobs through65537; no context reduction.
- `residual_norm_jobs.json`: public residual-input RMSNorm still needs separate residual output add. B1 0.557351, B32 0.669079, long 0.918009 ms. No material gain; no claim that an add was removed.
- `bf16_state_staged_jobs.json`: fixed the initial mixed-format kernel by expanding BF16 persistent state into FP32 scan CBs, retaining FP32 arithmetic and packing state only at the boundary. B6/B32 short pass (0.537660/0.656207 ms), but 1024-answer fails unchanged .995 gate at decode70/request8 (PCC0.993284). Independent quantized-state direct control queued to distinguish implementation from precision loss. FP32 stays default.
- `recurrent_identity_jobs.json`: inverse-identity elimination passes bitwise-state direct watcher, B32 0.666038 and long0.915496 ms, indistinguishable from default. Additional K*1 removal fails bitwise state control at B1 step1 (maxabs0.00012636), despite global PCC0.99999976; gate not waived. Inverse-only combined with fast zero fill queued because eliminating the reader bottleneck may expose compute savings.
- Both precise fused-gate candidates initially stopped at `ComputeConfigDescriptor(unpack_to_dest_mode=...)`: this field exists but is not constructor-bound. The supported writable-field assignment was verified with the installed binding, including propagation through KernelDescriptor. Exact snapshots and source/binding checks are in `packed_gates_audit/api_assignment_retry/`; fresh retry jobs preserve all original checks. Initial API errors are not performance or precision rejection evidence.
- Direct replicated-residual async AR BF8 payload candidates (persistent BF8 receive buffers, BF16 or BF8 output) queued separately from earlier hidden-residual BF8 family; unchanged BF16 layer-boundary residual and quality gates.

Commands use `source /home/ttuser/ornith-9b/lane-env.sh` then `python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.run_stage5_jobs <D>/<named_jobs.json>`; the runner records exact expanded commands, source snapshots, serialized device use, outcomes and recovery. No failed quality gate above is treated as passing.

### 2026-11-13 03:25 UTC — large-chunk results and remaining coherent families

The production source remains `5cdc9e59dc6c69e3f43284694ad4a225ac340f0c291f42060de94c8a156958a4`. No result in this entry is final default performance.

`dram_prefill_jobs.json` passed both layer kinds at all twelve logical boundaries around8K/16K/32K/64K, through65537. The large-norm path normalizes in DRAM, converts to BFP8, retains activations in DRAM above8192, and removes the forced projection-input L1 copy. With K16, warmed linear prefill atS16384/S32768/S65536 is26.912/53.732/106.959ms for8192chunks,24.312/50.987/102.141 for16384,24.134/48.095/101.416 for32768. Full attention is54.588/162.606/538.359,51.924/160.030/534.138,51.813/160.599/540.611 respectively. K8 alternatives pass but do not improve these results. Candidate choice: linear32768/full16384, subject to integrated max-context and watcher checks. No public alignment restriction and no context reduction.

B1 complete linear pools plus native batched full-attention routing pass: short0.510926ms, long0.546688ms, long minimumPCC0.999094. The matched existing private B1 long path is0.588612ms. Only pooled linear B1 history grows:393216 instead of12288bytes/layer/device, an additional380928bytes; across24linear layers,9142272bytes. The analytic all-layerB1 budget becomes17579245568 if selected. Independently allocated private state remains supported.

Fast-zero plus inverse-identity elimination passes independent bitwise-state watcher and1024-answer gates, long0.862244ms versus fast-zero-only0.863661ms: approximately0.16%, not a material gain by itself. Fast-zero wave64/80/86/96 long times0.890596/0.885952/0.866179/0.865396ms do not improve wave104's0.863661. Wave110 remains pending at this timestamp.

Replicated-residual BF8 direct-AR payload passes short and1024-answer gates at0.900902ms long; BF8 AR output plus BF16 residual conversion also passes at0.916619ms. These are compared with the same pre-fast-zero default0.915430ms, not a different residual topology. Direct BF8 row-projection output, both recorded-HF corpora, and persistent receive buffers are included in the next tests. The lower-movement hidden-residual families remain separately documented with stack-compatible consumption.

The precise packed-gate and conv-gate repair's supported writable API passed host binding checks but failed JIT: this build requires64 unpack-mode entries, not32. This is still an API adaptation, not a correctness or latency rejection. New helper variants and fresh retry labels preserve the failed evidence and avoid changing files used by queued jobs.

Additional outstanding families now have exact executable queues: current batched packed MLP (native sharded, interleaved split, and minimal fused SwiGLU with two K blocks); SDPA-only fidelity/approximation/destination precision with long context and recorded full-attention inputs; AR one/two-link, NoC-only and worker-placement choices; doubled recurrent input/output CBs with unchanged FP32 state and exact direct controls; BF16 persistent-state explicit RNE repair. The older packed MLP experiment only changed single-row `finish`, so it did not by itself reject packing on the current batch32 path.

Final validation infrastructure captures a separate watcher append-log byte range, hash and kernel map per run. `final_v1_*_jobs.json` are prepared, not executed; final source hashes, context, default chunk selection and final measured numbers remain pending. The time ledger will distinguish first-candidate wall interval, serialized experiment elapsed, CPU-only calibration and profiler postprocessing; none is described as accumulated kernel execution time.

### 2026-11-13 03:42 UTC — packed MLP, buffer overlap, and API adaptations

All25 jobs in `native_packed_mlp_jobs.json` pass their selected decoder/watcher gates. Native packed gate/up with8 DRAM readers is slower: B32short0.704116/long0.953590ms versus0.666156/0.915430 baseline. Two readers recover most overhead but remain0.669847/0.918427; interleaved slicing is slower still. Minimal fused SwiGLU K8 is0.885175/1.135015. Keep separate gate/up on the measured current path. This fills the earlier batch32 topology gap; one-token-only earlier tests are not used to reject it.

Both doubled-input and doubled-input/output recurrent CB candidates pass short/long/direct watcher. Long times0.863681 and0.863617ms versus fast-zero-only0.863661: no material gain for64/128KiB extra static CB storage per active core. Additional narrow-wave wraparound watcher jobs remain queued. Keep the smaller buffers unless later evidence contradicts this result.

The BF16 persistent-state RNE repair passes every direct production-wave case bitwise against explicit quantized FP32 control, but the narrow-wave probe retains old allocations until the next batch and hits L1 atB27. A new lifetime-clean probe preserves candidate L1 coverage and every gate, while reference-only buffers move to DRAM and all old views/temporaries are released. See`bf16_state_audit/lifetime_retry`. Independently, repaired BF16 long-answer decoder fails decode70/request8 PCC0.993030; shortB32 is0.669385, not faster thanFP32. Recorded-HF short/long controls are queued. No BF16 default selection.

AR one-link long0.950531ms, one-link NoC1-only0.981396, two-link NoC1-only0.926141, all slower than ordinary two-link0.915430. Corresponding64K128-step times3.761163/3.792616/3.737244ms versus3.726341. All these quality/watcher gates pass; watcher timings are not used for ranking. The fixed custom workers5,3/6,3 overlap the ordinary residual-grid reduction receivers; no performance rejection follows that compile error. New paired automatic/custom tests carry a shifted8x4 or compact4x8 residual grid through norm/adds and layer boundaries, with matching setup-owned receive buffers, rather than restoring the old layout immediately. Source proof and jobs: `ar_residual_grid_audit`, `ar_shifted8x4_retry_jobs.json`, `ar_compact4x8_retry_jobs.json`.

Further explicit adaptations: direct BF8 row projections now feed AR without a same-dtype typecast that incorrectly requests a simultaneous64→32-core change (`bf8_row_direct_jobs.json`). Precise gate helpers now use this build's64-entry unpack-mode vector (`precise_gate_repair_wide_modes_jobs.json`). New current-production hidden-residual families preallocate both norm-stat/activation AG and row-reduction outputs with separate role/layer banks (`current_hidden_residual_all_jobs.json`); they inherit current attention and tiled history and keep local1024 residuals across layers.

S1024 precision-matched kernel sweeps show legal2D prefill gains for packed input/output projections. `phase_prefill_jobs.json` tests complete layers at511/512/513/1023/1024/1025, input/output/both plus matched control, keeping direct fusedSwiGLU and unchanged dtypes/fidelities. `sdpa_geometry_jobs.json` crosses K64/256/512/1024 with current andHiFi2-exact FP32 policies on4K-long-answer and64K workloads; attention-table padding is owned internally, with cache-update tables untouched. `compile_args_jobs.json` directly tests the profiler's runtime→compile-time argument advice against fast-zero. Results remain pending at this entry; none is a final-default claim.

### 2026-11-13 04:00 UTC — precision and accounting follow-up

The SDPA-only precision sweep completed. HiFi2 exact/FP32, HiFi2 approximate/FP32,
and HiFi2 approximate/BF16 all pass the 4K/1024-answer and 64K/128-answer gates,
but save less than 0.4% at K128. LoFi exact/FP32 fails both long gates
(decode14 row28 PCC0.993746698; 64K decode2 row18 PCC0.994137049), despite short
recorded-HF success. These failures do not change the0.995 gate. K64/256/512/1024
geometry adaptations are queued before a final policy decision.

BF8 all-reduce payload with BF16 output passes recorded-HF short and long inputs
for both layer kinds. Long layer0 control0.396153182ms versus payload0.389243606ms;
layer3 control0.478819320ms versus payload0.471259123ms. BF8 output followed by
BF16 conversion gives0.401318436/0.474598296ms and is not the preferred family.
The BF8 row-producer API failure is not a rejection: the new direct-input variant
avoids a same-dtype typecast that accidentally requests a64-to32-core reshard.

`tests/summarize_stage5_time.py` now records interval-union accounting, separately
showing CPU calibration, hardware experiment elapsed, recovery/health work, and
profiler jobs (including host postprocessing). At03:59UTC,1119completed jobs cover
6.368h of serialized hardware+recovery intervals in a7.105h window. These are
experiment elapsed hours, not kernel execution hours; unfinished work is excluded.
The stop rule has not fired. Production remains unchanged at5cdc while the finite
queues complete; `final_integration_audit/base_integration_5cdc.patch` is unapplied.

### 2026-11-13 04:12 UTC — material fusion retries and BF8 producer result

The64-entry unpack-mode adaptation now compiles both fused gate families.
Packed-gate direct watcher passes, but whole-stack B32 short0.675083953ms and
4K/1024-answer0.925297639ms are slower than the5cdc control0.666155852/0.915430090.
The conv-wide precise family is faster (B32short0.646266551ms; long0.896500796ms),
but its legacy-convolution direct reference still fails B13/wave104/step5:
min-head PCC0.988250196. This is not treated as an acceptable numerical zero:
the failing head has RMS0.05556 and16.05% relative RMS error.

AutoFix splits the source investigation into convolution/reference equivalence
and gate arithmetic. The FP32 softplus kernel omitted INP_FLOAT32, selecting a
BF16-oriented polynomial despite FP32 DEST/unpack; production FP32-LHS TTNN
multiply+softplus sets the macro. New isolated fp32_softplus helpers add exactly
that compute define; their host binding/64-mode checks pass. Ten serialized
jobs now cover original and production-matched direct controls, both ordinary
and fast-zero cumulative paths, B6/B32, and the1024-answer gate. See
`conv_gates_g_audit/AUTOFIX.md`; no precision gate was weakened.

The repaired BF8 row-producer AR family passes its whole-layer and recorded-HF
short/long checks. It gives0.658284756ms B32short and0.907397131ms long, compared
with BF16 row output plus BF8 payload0.650869/0.900901598ms. Direct BF8 producer
output is therefore slower after a successful layout adaptation, not rejected
on its original API error. Larger recurrent buffers also pass wave7 watcher
coverage and still show no latency benefit; the smaller CB allocation stays
preferred.

### 2026-11-13 04:24 UTC — matched fusion control and fixture repair

The production-matched conv/gate direct watcher passes704records. Convolution,
history, and identical-gate recurrent/output controls are bitwise equal; separate
gate-preparation output min-head PCC is0.999977946. The legacy reference differs
at convolution rounding. A new production-versus-legacy control preserves the
legacy random fixture and strict head gate, to verify the original failure also
occurs with production convolution plus exact separate TTNN gates. The FP32
softplus macro repair and cumulative candidate comparisons remain queued.

The BF16 recurrent-state lifetime retry passes117.6s of direct watcher coverage,
including wave104 and wave7 boundary batches. The earlier L1 failure was probe
buffer lifetime, not an accepted hardware limitation. This does not rescue the
1024-step numerical failure; FP32 state remains selected.

The first phase-prefill comparison is INVALID reference evidence: reducing the
length list changes its maximum, capacity, and the initial torch.randperm RNG
consumption. Even control failed first511-row PCC near0.2157. Host SHA proof is
`phase_prefill_fixture_audit.json` (seed73; capacities1120vs16480). Fresh baseline
references with the exact six-length list, then all eight candidates, are queued
in `phase_prefill_matched_jobs.json`. No option is rejected on this fixture error.
The unlaunched final large-prefill watcher lists now match the full12-length
reference lists, including max65537 and steps4.

### 2026-11-13 04:34 UTC — current residual families and dispatch repair

The updated carried-hidden residual family passes whole-stack gates: separate
B32 short/long 0.965026859/1.206143433 ms, persistent all outputs
0.927936099/1.167758205 ms; persistent BF8 short 0.910134171 ms. Fused
scatter with output buffers passes at 1.998956245 ms. These paths carry
width1024/rank between layers, with no immediate replicated-contract restore.
The current replicated control is 0.666155852/0.915430090 ms.

Fused gather initially failed K16384 versus K_w4096 during scalar continuation:
generic norm had already gathered width4096, and decode_linear gathered again.
AutoFix's new dispatch wrapper bypasses fused gather only for already-full-width
packed_input, preserving packed B1/B32 width1024 fusion and scalar fused scatter.
Thirty host dispatch/bank checks pass. Five serialized retry jobs cover B1,
B32 short/long, gather-only, and watcher. See
`current_hidden_residual_audit/dispatch_retry/AUTOFIX.md`.

Moving four fixed runtime words per core into compile arguments passes short,
1024-answer and direct watcher gates. Long latency 0.863355864 versus fast-zero
0.863660616 ms is only 0.035% better, not a meaningful reason to retain another
kernel variant. This directly tests the profiler advice concerning runtime args;
decode is already traced.

The shifted8x4 residual grid resolves the original worker-core overlap and passes
both automatic/custom workers, B1/B32 short, 64K, 1024-answer, and watcher.
Automatic/custom long values are 0.914283097/0.912395003 ms versus 0.915430090 ms
control; 64K 3.723879205/3.721449105 versus 3.726341086 ms. These small deltas
are recorded as successful adaptation evidence, not a rejection at the first API
error. Compact4x8 remains in the serialized queue.

### 2026-11-13 04:37 UTC — collective placement retries complete

Both worker-disjoint residual grids now have successful whole-family evidence:
20 jobs passed across automatic/custom AR workers, B1/B32,64K,1024-answer,
and watcher. Compact4x8 automatic/custom long latency is
0.921015954/0.920231361 ms;64K3.731385618/3.730400279 ms. It is slower than
the origin8x4 control. Shifted8x4 custom is only0.33% faster long and0.13%
faster64K, so origin8x4 remains the simpler preferred grid pending final
cumulative selection. No API-error-only rejection remains for this family.
Prepared final watcher job lists use TT_METAL_WATCHER=10, NOINLINE=1, with
assertions enabled and separate profiler jobs.

### 2026-11-13 04:50 UTC — SDPA adaptation and final fixture coverage

The reduced-destination SDPA geometry retry passes key chunks64/256/512 on
1024-answer and64K. Exact measured files/ms:

- `sdpa_geometry_hifi2_approx_bf16_k256_sdpa_precision_hifi2_approx_bf16_answer1024_32.json`: 0.919008604 ms.
- `sdpa_geometry_hifi2_approx_bf16_k256_sdpa_precision_hifi2_approx_bf16_context64k_32.json`: 3.724405193 ms.
- `sdpa_geometry_hifi2_approx_bf16_k512_sdpa_precision_hifi2_approx_bf16_answer1024_32.json`: 0.943420711 ms.
- `sdpa_geometry_hifi2_approx_bf16_k512_sdpa_precision_hifi2_approx_bf16_context64k_32.json`: 3.733787593 ms.
- `sdpa_geometry_hifi2_approx_bf16_k64_sdpa_precision_hifi2_approx_bf16_answer1024_32.json`: 0.907756109 ms.
- `sdpa_geometry_hifi2_approx_bf16_k64_sdpa_precision_hifi2_approx_bf16_context64k_32.json`: 3.714476712 ms.

K1024 still fails after precision adaptation: static CB end1527616 exceeds the
first live L1 buffer1432704 by94912 bytes on scalar continuation's8x8 grid.
This is a limit of the measured resident family, not a claim that all possible
L1 lifetimes/layouts are impossible. Smaller legal chunks run successfully;
K64 remains the best measured geometry. No public input-alignment requirement
was introduced, and cache-update tables are separate from attention padding.

The128K-reference job passes383.6s: B32,131071prefix,132128capacity,1024steps,
with state checks. The final TP4 job remains unlaunched until integration.
Read-only final fixture audit confirms matched RNG/capacity, including the
four-step profile's explicit5152capacity; the1024-step reference's terminal
state must not be compared after four steps. Base integration repairs the probe's
forced1024chunk override. Additional immutable stage4 and final-default clean
prefill measurements now cover511/512/513/1023/1024/1025 and twelve large
boundaries through65537, with three warmups/five repeats. These supplement the
large-prefill watcher correctness runs, whose timings are not ranked.

### 2026-11-13 04:54 UTC — FP32 softplus repair proves exact current arithmetic

`conv_gates_fp32_softplus_matched_direct_watcher.json` passes704records in86.3s.
All records have bitwise-equal production convolution/history, TTNN beta/g,
identical-gate state/output, and independently prepared separate-gate
production state/output. The INP_FLOAT32 repair eliminates the earlier gate
polynomial difference exactly on this fixture. Both legacy direct probes still
fail the sameB13/wave104/step5 head PCC0.988250196; g/beta are now bitwise equal,
while legacy separate convolution differs. The paired production-versus-legacy
control remains queued and must classify that discrepancy before final selection.
Fast-zero plus repaired conv/gates gives B32short0.599253946ms and
1024-answer0.849053101ms, versus fast-zero-only0.614738604/0.863660616ms.
The ordinary repaired conv/gates path gives0.648132269/0.897504040ms.
No final-default claim yet; combined BF8-payload/SDPA tests remain queued.

### 2026-11-13 05:06 UTC — cumulative selection and repaired topology family

All20 combined decode jobs pass. Family order is zero-fill/pool-B1 control,
plusBF8 AR payload, plusHiFi2-exact-FP32 SDPA K64, conv+BF8, conv+BF8+SDPA.
B32 long values are0.864001154,0.852527,0.844115,0.835905,0.828452804ms;
final candidate B1short0.483731064,B32short0.587295205,64K3.634062409ms.
Select the complete conv+BF8+SDPA family for final-default integration/gates.
Residual/outputBF16, recurrentFP32 and exactHiFi4 GDN remain unchanged.

Both repaired conv/gate production-matched watchers pass704records, with bitwise
convolution, gates, histories, states and outputs. The production-versus-legacy
control reproduces the sameB13/wave104/step5 head PCC0.98825019598. CPU comparison
proves both actual and expected failure tensors are bitwise equal across ordinary
fusion, fast-zero fusion, and production-substitution controls. See
`conv_gates_current_audit/anomaly_resolution.json`. This is controlled reference
arithmetic, not a lowered gate; final integrated direct coverage expands to all32
batches and both104/7 waves.

All eight matched phase-prefill candidates pass the exact six-length fixture.
AtS1024 control/input/output/both: linear2.0330/1.9809/1.9853/1.9404ms;
full2.1416/2.1619/2.1116/2.0919ms. Select both input/output2D geometries for
512..1024 operation rows, with unchanged fusedSwiGLU and dtypes. Outside that
range keep the measured minimal family. Final short-boundary watcher coverage
was added. Existing final large_boundaries jobs already measure clean warmed
prefill; avoid duplicate final large runs and add only matched frozenstage4
before rows plus final short-boundary perf rows. The stage4-prefill queue's copied
`purpose` string is editorially stale; its actual command/wrapper/result SHA
identify the immutable TP4 stage4 implementation and explicit chunk1024.

The fused carried-hidden dispatch repair passesB1/B32,1024answer,gather-only,
and watcher. B32short3.007805394,long3.257096093,B1short2.986786840ms;
gather-onlyB32short1.956843492ms. The family is slower after successful adaptation,
without any old replicated-contract restore at layer boundaries. MixedBF8 row
input/fused collective compatibility remains queued for completion.

### 2026-11-13 05:15 UTC — final default integrated and validation launched

Final production SHA256:
`80ad61a028434db6d7ebaa46f0e051ef375b09f3246aca0126b50fbdcfb523e9`.
Applied the base integration patch and optional exact conv/gate fusion; integrated
BF8 receive buffers/payload with BF16 AR output, separate SDPA-only HiFi2 exact
FP32 config/K64, and measured2D input/output prefill projections for512..1024
operation rows. Default chunks32768linear/16384full are guarded to replicated,
nonfused prefill; explicit overrides and other modes keep their contracts.
The probe omits unspecified chunk overrides and records effective chunk sizes.

All17 selected production/final-harness Python files pass configured Ruff and
Black(py312), AST parsing;19 local C++/header files were clang-formatted;
`git diff --check` passes. See `final_style_validation.json`. No frozen reference
or optimized single-chip implementation was edited. Context accounting records
pooled-B1 history increase9,142,272bytes across24linear layers and shared BF8 AR
scratch1,114,112bytes/chip. Supported context remains262144, with final validation
explicitly pending; no all-layer allocation claim.

All finite candidate queues completed. Background sweep stopped between configs
viaSTOP, closed cleanly and passed its recovery smoke; no live job was killed.
`final_v1_all_jobs.json` launches74 jobs under one lock and stops on any failure:
actual-production direct watchers,55 decoder gates,8recorded-HF checks, clean
S1024/short-prefill timings,128K1024-answer stress, and separate final profiles.
Final numbers will be taken only from this default. The time ledger before
validation remained below8h of serialized device work; whole-target70% was not
claimed. Stage review and local commits remain pending.

### 2026-11-13 05:20 UTC — actual-production direct fusion gate passed

`final_conv_gates_production_direct_watcher_all_batches.json` passes222.53s,
2048rank records: allB1..32 ×waves104/7 ×8steps ×4chips. Every convolution,
beta/g, history, state and output comparison against independently prepared
current-production controls is bitwise equal. The run imports actual integrated
helpers and asserts the decoder binds those same functions. Watcher append-range
capture and source hashes are in its device-job/provenance sidecars. Remaining
final decoder gates/profiles are still pending.

The final mixed-BF8 fused-collective compatibility retry also passes all three
jobs: B32short2.501782752ms,long2.751041646ms,watcher. BF8 row inputs improve
this fused hidden family relative to itsBF16 version3.007805/3.257096ms, but it
remains slower than the replicated-residual family. The adaptation retains
width1024/rank across layer boundaries and all supported persistent outputs.

### 2026-11-13 05:24 UTC — final primary workload reproduces selection

All three actual-production direct watcher jobs pass. The integrated main target
`final_v1_answer1024_32.json` passes1024evolving steps/state checks under tracking1,
skip-program-cache0: minimumPCC0.9960771799; traced warmed latency0.829040562ms.
This reproduces selected candidate0.828452804ms within0.071%. The metric is from
production80ad61a, including selected prefill defaults and phase kernels. Remaining
64K/max-context/pool/recorded/profile gates are running; no completion claim.

### 2026-11-13 05:27 UTC — final context contract preserved

Final B32/64K128-step gate passes at3.634035750ms. Both final max-context
jobs pass full logical prefill262139,262143,262144 and decode position262143:
linear24.78s/full46.69s job elapsed. Context metadata now binds those artifacts
to80ad61a and retains262144 without reduction. The new B1 all-layer budget is
17,579,245,568bytes/chip (analytic, including pooled history), not an all-layer
runtime claim. Shared BF8 CCL scratch is L1; its reduction was not subtracted
from the conservative DRAM capacity budget. Default recurrent backing remains
FP32 DRAM; recurrent-write shards are temporary, not all-layer L1 residency.

### 2026-11-13 05:35 UTC — final watcher/contracts and B1 regression rows

Both large-prefill watcher runs pass using actual omitted defaults32768linear /
16384full. Both refresh/positive-control contracts and128-step stress pass.
The FUNCTIONAL bandwidth-floor fractions atB1,128in/128validated-out are0.40431
(linear,0.290311100ms) and0.41491(full,0.262635462ms), above0.10. Final packedB1
short decode0.483557954ms improves the matched stage4 value0.578763057ms.
The batch matrix continues; no final review yet.

### 2026-11-13 05:39 UTC — harness setup cleanup and repeated timing

A self-check found unused per-request zero input allocations in the packed-I/O
harness (the packed input is the only used decode input). Ruff's unused-binding
fix had left the allocating expression in place. Stopped the queue between jobs,
removed that expression, passed Ruff/Black, and resumed remaining jobs with
`final_v1_resume_setup_clean_jobs.json`. No production source, RNG draw, trace
body, threshold or reference changed. Completed correctness evidence remains
bound to production80ad61a and exact per-job harness snapshots.

Repeated primary/B1/64K timing labels use `_setup_clean`; unrun B31/B32 and later
jobs use the cleaned harness directly. Primary final selection is now
`final_v1_answer1024_32_setup_clean.json`:0.828550360ms, all1024-step/state gates
pass. The original0.829040562ms result is retained as historical evidence, not
the selected final number. The final manifest lists the repeated timing rows.

### 2026-11-13 05:40 UTC — optimization time box reached

`stop_rule.json` freezes the ledger at05:40:28UTC: hardware-job intervals7.245903h,
recovery/health0.792634h, overlap-deduplicated union8.001707h from the first
candidate20:52:36UTC. Wall window8.797562h; host-only calibration0.026266h is
excluded. Profiler-job elapsed0.466970h is included and separately identified;
setup/compilation/validation/teardown and profiler host processing are not called
kernel execution time. The user-selected eight-hour device-work rule fires;
whole-target70% is not claimed. Search ends, mandatory quality/profile/review and
local commit work continue. Every batch1..32 has passed final default checks.

## Not tried (stop rule)

- Remaining exhaustive prefill geometry combinations beyond measured role/precision winners. Expected gain: unknown; representative winners were crossed into whole layers. Evidence/context: prefill_phase_geometry_specs.json and per-role sweep JSONs; STOP terminated between configurations.
- Separate attention input projections versus the final packed B32 family. Expected gain: unknown; packed prior-stage path preserved; native extraction/partial norm-RoPE already fused. Evidence/context: Initial topology audit and prior optimized packed path; no current-stage measured rejection claimed.
- Custom dual-output residual-add plus RMSNorm kernel. Expected gain: at most a few elementwise launches, not measured. Evidence/context: Public residual-input RMSNorm tested but still required the separate residual add and did not win.
- Further K1024 SDPA L1 lifetime/residency restructuring. Expected gain: unknown; K512 was slower and K1024 destination-precision retry retained94912byte overlap. Evidence/context: sdpa_geometry_jobs.json and sdpa_geometry_bf16_jobs.json; not a global impossibility claim.
- Fully BF8-normalized fused AG+matmul staging beyond the tested mixed BF8 fused-row family. Expected gain: unknown; measured mixed fused family remains substantially slower. Evidence/context: hidden_dispatch_mixed_jobs.json; BF8 separate/persistent norm/AG tested.
- Further merging native QKV extraction with partial norm/RoPE. Expected gain: unknown; existing native extraction and fused partial norm/RoPE measured separately. Evidence/context: Final profile will bound remaining operation cost.

### 2026-11-13 05:47 UTC — all final decoder gates pass

All55 selected final decoder gate jobs pass, including everyB1..32, both kinds'
contracts/positive control/stress/max context, long-answer/context, ordered and
subset/permuted/mixed pool/reset/isolation checks, and all watcher jobs. Clean
large-prefill at8192/16384/32768/65536 is12.850/24.207/48.051/101.349ms linear
versus17.612/34.637/69.462/142.235ms frozenstage4; full19.929/52.143/159.907/
532.448ms versus24.785/62.647/177.808/577.608ms. Exact sample vectors and PCC
are retained. Recorded-HF checks, additional128K stress and final profiles remain
in the queue. No correctness gate has been waived by the time box.

### 2026-11-13 05:49 UTC — final recorded-HF evidence passes

All eight recorded-input jobs pass (B1/B32 ×linear/full ×short/long),
including residual-subtracted contribution and replica-agreement gates.
- `final_v1_recorded_1_0_short.json`: minimum PCC 0.999947071.
- `final_v1_recorded_1_0_long.json`: minimum PCC 0.999926507.
- `final_v1_recorded_1_3_short.json`: minimum PCC 0.999120176.
- `final_v1_recorded_1_3_long.json`: minimum PCC 0.999393761.
- `final_v1_recorded_32_0_short.json`: minimum PCC 0.999921918.
- `final_v1_recorded_32_0_long.json`: minimum PCC 0.999898672.
- `final_v1_recorded_32_3_short.json`: minimum PCC 0.998905659.
- `final_v1_recorded_32_3_long.json`: minimum PCC 0.999140680.

### 2026-11-13 05:51 UTC — clean per-kind timing complete

FinalS1024 warmed prefill/traced decode: linear1.938097179/0.290193241ms;
full2.101829741/0.275786957ms. Before2.089963/0.307854 and2.189704/0.288252ms.
These use the private single-request per-layer API; the packed B1/B32 stack
rows separately measure the pooled/native batch API. Both are realTP4. Final
short-boundary perf rows also pass;128K1024-step stress is in progress.

### 2026-11-13T05:58:08.591712+00:00 — 128K long-answer accuracy investigation

Final default 80ad61a passed all55 standard gates, all8 recorded-HF gates and all3 direct-kernel watcher jobs. Additional B32/prefix131071/capacity132128/1024-step test failed at decode_6_11, PCC0.9914271235 vs0.995. Ranks agreed; clean device close and recovery mesh smoke passed, no reset. This is required work, not waived by the timebox. AutoDebug launched inspection-only (namespace preflight failed; explicit no-additional-sandbox retry under existing unrestricted authorization). AutoFix controls preserve same seed/page-table capacity: first8 decode steps with chunk1024, SDPAHiFi4/K128, and frozenstage4. Removed terminal state checks only for these truncated diagnostic runs because their reference is after1024steps. Added failure-only tensor/PCC-history artifact capture to harness without changing computations or gates. Final profile queue remains unrun.

### 2026-11-13T06:08:04.958059+00:00 — SDPA discrimination

HiFi4/K128 passes all first8steps. HiFi2/K128 passes through step1012 but fails step1013,row12 PCC0.9946861863; minimum earlier decode PCC0.995606482. Thus8steps was insufficient to select a repair; full1024-step gates retained. HiFi4/K64 and frozenstage4 workspace-corrected controls are running, then HiFi2/K256 and K512 with owned table alignment. Stage4 initial control was INVALID: current BF8 collective scratch was inadvertently passed to frozen BF16 reduction (32768-byte CB vs17408-byte bank). Harness now uses the frozen collective workspace for --stage4; no production mutation. AutoDebug report identifies BF16 SDPA running statistics and683updates/worker at128K/K64 versus32–33 in the singlechip reference, not a proven indexing defect.

### 2026-11-13T06:16:18.667512+00:00 — first full 128K repair candidate passes

HiFi4/K64 fails decode1,row16 PCC0.9945239425, rejecting fidelity alone. Correct frozenstage4 workspace passes the8-step original-query control; full1024-step baseline queued. HiFi2/K256 with internally padded attention tables passes all1024steps and final history/recurrent/KV checks: minimumPCC0.9957183599 atdecode1013,row12, traced warmed6.746857846ms. K512 fullrun pending. AutoDebug completed; final report preserved in context128k_autofix/AUTODEBUG.md. Supported cause is SDPA chunk-policy numerical exposure; precise per-op arithmetic attribution remains inferential without identical-operand replay. Proposed capacity-based K selection bounds the number of BF16 online-softmax updates/worker, retains K64 at4K/B32, and owns table padding before pool offsets. No production change yet.

### 2026-11-13T06:24:09.673093+00:00 — adaptive SDPA integrated; final82-job queue

Production SHA256`b1d5016dd769ab65fce82e27d483f73a33fa2e746e29e21952d521052948d2d8`. Select the smallestK64/128/256/512 bounding local BF16 running-statistic updates to200 perworker across the original table capacity; explicit max16workers/head matches selection. Scalar64cores, native110cores; no host position readback. Request-local attention-table padding precedes pool offsets, update tables unchanged. K256/128K full1024PCC0.99571836,6.746858ms; K512PCC0.99641383,6.763172ms (0.24%slower). Frozenstage4 full1024 passes minimum0.99865484 at28.897929ms. Select smallest safe measured chunk;200 is a conservative numerical-risk bound, not a mathematical PCC guarantee. Host exhaustive270336 scheduling/capacity checks pass. All final gates/timings/profiles rerun asfinal_v2 on current default;8 extra watcher cases cover128K and both sides of K transitions with valid unused virtual pages aliasing local physical page0. These short-position table tests do not claim populated long-context KV coverage. No full-model work.

### 2026-11-13T06:28:54.160596+00:00 — final default 128K passes; transition fixture repaired

Finalb1d5016 passes full128K/B32/1024steps plus terminalstate, minimumPCC0.9957183599,6.734446506ms versus stage4 28.897929005ms. Additional virtual-tail fixture INVALID: paged_fused_update_cache requires table width<=physical cache blocks (1200>96). Adapted the test to allocate true matching capacity; table permutations map distinct valid pages. Additional large-capacity tests explicitly sample first/last and valid-prefix physical pages to bound host transfer, while ordinary pool tests retain complete-state comparisons. This covers K/padding/control-flow transitions at short positions, not long-populated-KV numerical coverage (separate128K test). No production change. Failed artifact retained; fresh capacity_extra labels resume remaining81jobs.

### 2026-11-13T06:33:55.878396+00:00 — capacity-transition watcher coverage passes

All7 corrected capacity_extra watcher jobs pass:1197/1198,2397/2398,4797/4798 and8187 extra pages, exercising both sides of K64→128→256→512 and largest8192-page heterogeneous table. Allocated contexts match tables; ordered32, reversed32, subset3, single1, mixed5 paths and tracked interleaved traces execute. These tests use sampled state pages by explicit metadata; ordinary pool tests remain complete-state. Production SHA unchangedb1d5016. Populated128K watcher and original final matrix continue.

### 2026-11-13T06:44:48.710400+00:00 — populated128K watcher and fused-kernel gate pass

The populated128K/B32 watcher run completed cleanly in523.35s. During its longer instrumented prefill, live watcher host IDs advanced13453→16166, ruling out a suspected stall; no triage/reset/termination was needed. Actual production conv/gate direct watcher passed2048records again under b1d5016, followed by standalone recurrent watcher gates in the queue.

### 2026-11-13T06:52:25.259800+00:00 — final primary,64K,HF and max-context gates pass

All8 recorded-HF gates pass on b1d5016. Final4K/B32/1024step traced warmed0.8285623975ms, minimumPCC0.9960771799; stage4before19.8579785414ms. Final64K/B32/128steps3.6345340544ms, minimumPCC0.9977274537. Bothkind complete max-context gates pass262139/262143/262144 with terminaldecode262143; context contract updated to current-source passing evidence, remaining final gates/profiles/review pending.

### 2026-11-13T07:28:07.542397+00:00 — all82 final jobs, profiles and consolidated audit pass

Finalb1d5016 source is bound in final_manifest.json (79artifacts+3direct-kernel artifacts). audit_stage5_evidence passes82artifacts, minimumPCC0.9957183599; fallback audit clean. summarize_stage5_final emits45matched timing rows and bothFUNCTIONAL bars40.4/41.5%. FinalS1024 prefill1.950112/2.096221ms vs2.089963/2.189704; private traceddecode0.290313/0.275899ms vs0.307854/0.288252. FinalB1/B32 shortpacked0.483737/0.587548ms. PrimaryB32/4K/1024after0.828562ms vs19.857979before.64K3.634534ms vs24.223269;128K6.734447ms vs28.897929.

All5final advice-enabled tt-perf-report tables/CSVs/provenance exist. Target55ops/rank:771.478us mergedkernel+46.060us gaps; ordinaryops take maxrank, CCLmeanrank. Independent perchipFWspan maximum812.335us atcaptured1350MHz, same-runhost880.320us. Same-runpositions4115–4118 floor0.430016ms; primarylong-endpositions5135–5138 have separate0.464832ms floor. Runtimeattributes proveBF16/BFP8/BF16 and4HiFi2+6LoFi matmuls,2workers/bank input/gate/up and4output/down. Corrected topology wording that confused Cores=8 with workers/bank; rawmetadata is preserved in selected_runtime_rows.json. Generic toolFLOPs>100 and27.1% modeled-ops DRAM number are not wholepath utilization claims. Finaladvice.md maps all advice to measurements or stop-rule handoff.

Stop-rule point remains frozen at8.001707hardware/recovery hours. Final ledger including mandatoryaccuracyrepair/revalidation reports9.384609hours across1436jobs, with0.570186profile-job hours including host postprocessing; not kernel time. Independentreview and localcommits stillpending.

## Not tried (stop rule): final-profile follow-ups

- Extend compile-time argument specialization to the final fused convolution/recurrent helper. Earlier fast-zero helper trial saved0.035%; finalreport estimates atmost3us gap reduction. Not a measured rejection of the final helper.
- Pad S1024 projection M/N to occupy every configured core beyond88linear/80full activecores. Precision-matched geometry was swept; this additional padding variant was not measured. Ideal gain beforepadding cost29–35us/layer; actual gain unknown.
- Continue exhaustive large-prefill MinimalMatmul geometry after the tested configurations. Toolgeneric no-program-config advice does not recognize explicit MinimalMatmulConfig.
- Explore FP32 SDPA running statistics as a separate TTNN kernel change. The capacity-aware model-local repair passes current unchanged gates; broader numerical robustness/performance benefit is unmeasured.

### 2026-11-13T07:43:58.379073+00:00 — independent review round 1; timing remediation

[stage_review_round1.md](stage_review_round1.md) returns more-work-needed for R1 only: separate diagnostics-off performance with matched settings. All82 quality artifacts,200526PCC values,26production hashes,20watcher captures and5profile CSV hashes independently pass. Retain alltracked gates and profiles. Run17fresh serialized timing jobs in review_timing_jobs.json withtracking/tracebacks=0, watcher/profiler unset, unchangedsource/workloads/gates. review_timing_map.json links superseded timing only. No decoder edit or fullquality-matrix rerun required. Fix summary acceptance to reject diagnostics or setting mismatch. Earlier short-prefill evidence is accepted; add index mapping and align context top-level evidence.

### 2026-11-13T08:05:56.184826+00:00 — R1 remediated: 24 clean timing runs pass

All17 initial and7 supplemental timing jobs pass, with tracking/tracebacks=0 and explicit watcher/profiler/slow-dispatch unsets. The7 supplemental jobs replace even the previously clean rows, so every final comparison has identical explicit settings. Production remains b1d5016. `stage5_timing_evidence.py` checks provenance/current production hashes, frozen-before role, PCC, completed jobs, diagnostic settings and pair workload/reference/warmup/harness equality.11 host positive/negative controls pass. `summarize_stage5_final` emits45matched rows; final_timing_audit.json passes24artifacts. Original82 quality jobs and5profile tables are retained and re-audit clean.

- packed_stack_B32_4Kprefix_1024answer_decode: 19.855421339 → 0.828630803 ms (23.9617×).
- packed_stack_B32_prefix65535_answer128_decode: 24.221872608 → 3.635507589 ms (6.6626×).
- packed_stack_B32_prefix131071_answer1024_decode: 28.889897815 → 6.731745554 ms (4.2916×).
- layer0_S1024_prefill: 2.098790836 → 1.963015180 ms (1.0692×).
- layer3_S1024_prefill: 2.215763088 → 2.073185053 ms (1.0688×).

Final FUNCTIONAL fractions40.419/41.516%, primary roofline56.096%. Measured small-prefill regressions remain visible: linearS5132.22%, fullS5124.03%, fullS5130.24%; no universal improvement claim. Top-level context evidence now references v2, retained_short_prefill_evidence.json indexes accepted short-boundary controls. Device ledger1460jobs,9.715293hardware/recovery hours; frozen optimization stop remains8.001707hours. No further optimization search or production changes.

Reproduction: run `tests.run_stage5_final_jobs` on `review_timing_jobs.json` then `review_timing_additional_jobs.json` (fresh labels required for reruns); run `tests.prepare_stage5_manifest --collection final_v2`, `tests.audit_stage5_evidence`, `tests.summarize_stage5_final`, `tests.index_stage5_evidence`, `tests.summarize_stage5_time`, under the model module prefix. Follow-up independent review pending.

### 2026-11-13T08:10:15.099609+00:00 — checkpoint validation before follow-up review

Black(py312), Ruff and diff checks pass for the3 changed host-summary/index files (`review_remediation_style.json`). Production kernel CB-pointer and shared-helper checks pass (`production_kernel_host_checks.json`). Repository pre-commit checks pass with explicit local-checkpoint exceptions: `SKIP=trailing-whitespace,end-of-file-fixer,black,autoflake,isort,clang-format,check-large-files`. These exceptions preserve byte-for-byte measured source/snapshots and archived raw evidence, including files above500KB. Production Python/C++ style evidence is already retained; no accuracy/runtime gate is skipped. All remaining applicable hooks, including merge-conflict, API, global-Torch-import, CB-pointer and shared-helper checks, pass (`checkpoint_hooks.log`). No hook mutated source. This is a local evidence checkpoint, never a push.

### 2026-11-13T08:20:28.997707+00:00 — independent review round 2 clean-pass

[stage_review_round2.md](stage_review_round2.md) returns **clean-pass**, with no Required Work. R1 is closed. The fresh reviewer independently checked24 successful nonoverlapping diagnostics-off timing jobs,145540PCC values,166timing medians,45comparison rows across11pairs,26production hashes and20negative controls. No more device runs or review rounds are required. Empirical SDPA margin, populated-B32-max-context/all-layer limits, measured short-prefill regressions, stop-rule handoffs and checkpoint-hook exceptions remain explicitly documented. Only local checkpoint creation/SHA receipt follows; no full-model/vLLM work and no push.

### 2026-11-13T08:22:22.136208+00:00 — local checkpoint and completed stage

- Repository: `/home/ttuser/ornith-9b/tt-metal`.
- Branch: `main`.
- Stage implementation/evidence commit: `1b91dfedb8102f44c3b7a5036f6e81782260260f` (`Optimize Ornith 1.5 9B TP4 decoder and preserve stage evidence`).
- Review: [round2 clean-pass](stage_review_round2.md); no Required Work.
- Commit hooks: all applicable non-format checks passed with the byte-preserving/local-evidence exceptions recorded above.
- Worktree was clean immediately after the checkpoint. This SHA/status receipt is a separate documentation-only commit; it does not alter measured source, artifacts or gates. Its commit is available in the local Git log and final stage response.
- Push state: never pushed. Only tt-metal was changed. No full-model or vLLM work began.

Stage5 `optimized-multichip-decoder` is complete. Final performance comes from24 diagnostics-off timing runs; all82 retained quality jobs,5profile reports, source/fallback/context/watcher gates and independent review pass. Stop-rule handoffs and measured per-shape tradeoffs remain documented.
