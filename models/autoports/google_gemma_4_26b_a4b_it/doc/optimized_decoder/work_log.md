# Stage 03 optimization work log

Status: in progress. Starting commit b36116ff5c, clean working tree.
Target google/gemma-4-26B-A4B-it revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52.
Scope: optimized_decoder.py, tests and documentation only.

Startup: installed optimize/device-usage and repo-local equivalents read; installed
AutoDebug dependency exposed in active skill inventory and environment.py validation
passes. LLM report section 4 read. `timeout 60 tt-smi -ls --local` exits 0,
four P300c Blackhole ASICs (two boards). Single MeshShape(1,1) open/close exits 0
and prints OPTIMIZE_MESH_SMOKE_OK. Firmware 19.9.0, KMD 2.10.0, 110 workers;
AICLK settles at 1337MHz versus requested1350 (within runtime tolerance).
No stale experiment, reset, watcher or profiler process. Watcher/profiling remain
separate serialized runs. Existing fused stage evidence is the baseline, not
optimized-stage evidence.

## Initial measured topology audit

Baseline: fused_decoder/tracy/{sliding,full}_verified/*_perf_report.{txt,csv},
whole_layer.json and fused_decoder/PERFORMANCE.md. Required workload 4096/128/B1/C1.
Sliding/full whole-layer device prefill 2812752.09/2825397.16us;
decode 5146.70/5584.81us. Correct host candidate controls 5067.23/5521.44us.

| Current operations | Candidate replacement | Constraints | Action/evidence |
| --- | --- | --- | --- |
| FP32 QKV broadcast multiply/reduce, packed/tied weights | native projection, compensated decomposition, DRAM sharding | FP32 activation precision and close router ranks; .995 output PCC | AutoDebug source investigation; no candidate accepted |
| Packed sparse expert gate/up, slices, GELU/mul, sparse down | reduced weights/fidelity crossed with K blocks/grid; tuned separate gate/up | active top8, exact nnz, BF16 output, L1 intermediates | precision/config sweep next |
| Shared packed gate/up, GELU/mul, sharded down | per-role BFP4/BFP8, separate activation fusion, DRAM-sharded decode | preserve downstream norm and output PCC | pending |
| FP32 input/post/head norms and residual traffic | width-sharded working residual and explicit norm configs | sliding precise heads previously required | pending |
| Precise paged attention/cache gather and untilize | native SDPA, reduced cache, tiled gather | previous adapted native SDPA fails .995; revisit optimized compatible policies | existing evidence inspected; optimization trials pending |
| Dense all-expert prefill per64 tokens | routing-driven active prefill sparsity / larger legal sparse batches | per-token routes and nnz invariant; logical tails | pending |
| No CCL, LM head, sampling | not applicable single-device layer | later stages excluded | no later-stage work |

Inherited context262144 and arbitrary logical lengths remain unchanged.

## Initial experiments

`python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_decoder`
with real weights, layer0, logical length33,8 traced steps:
- --expert-gate-dtype bfloat4_b (both phases, initial implementation): prefill
  PCC .99439798; decode minimum .99335720. Deterministic, runtime audit clean;
  exit1. expert_bfp4_gate_sliding33.{json,log}. Not accepted. Subsequent source
  separates prefill source from decode weights; these older results are both-phase.
- --compensated-qkv, BF16 expert controls: prefill .99823577 and decode minimum
  .99966628, deterministic, exit0. compensated_sliding33.{json,log}.
- Same compensated candidate at4096/128 with --timing --verify-program-cache:
  prefill .99855290; only position4149 fails (.99327746); exit1.
  Host traced median3404.36us, not device time. compensated_sliding4096.{json,log}.
  Do not accept performance before correctness is repaired. Fresh AutoFix
  subagent investigates FP32 projection decomposition/accumulation; diagnostic
  runner failed due missing bwrap, source-only subagent inspection continues.

The first three result files label decoder='fused' because the unmodified baseline
harness was injected with OptimizedDecoder. They are candidate evidence only;
commands above identify actual path. The wrapper now records decoder='optimized',
complete candidate arguments and runtime source SHA256 even on a failed gate.
No old result is relabeled as passing or used for final acceptance.

## First whole-layer profile

Combined active-prefill/grid44 K11 and BFP8/LoFi grid44 K11 decode passes both
real4096/128 kinds: sliding minimum decode .99891929, full .99963133.
Sliding profile exits0; native firmware complete-window accounting gives
prefill549466.15us versus fused2812752.09us, decode4581.62us versus5146.70us.
These are candidate results, not a frozen final default. Host decode is4496.45us
in the corresponding unprofiled candidate; profiling adds overhead and these
are separate runs. Runtime sparse decode rows prove BF16 x BFP8, LoFi, K11.
Prefill uses BF16 source weights, HiFi4 and active union with inferred nnz.
The common summarizer's final precision assumption is inherited and stale
(BF16 everywhere); replace that wording before final report/telemetry.

Advice-enabled reports live in tracy/expert_candidate. Prefill's initial
--active-experts8 option understates union-of64-token work: its utilization
percentages are not valid for prefill sparse rows. Complete-layer useful FLOPs
still count actual model top8 work; device time remains the entire window.
Dominant decode: QKV SFPU multiply2279us+reduce252us, precise attention matmul192us,
cache untilizes~207us, sparse gate/up97us. Remaining advice includes sharded
norm/residual, DRAM-sharded dense projections and expert per-role geometry.

A precision-locked expert sweep now crosses BF8/BF4 gate-up, BF8 down,
LoFi/HiFi2,11/22/44 workers and K2/11/22. Each case uses real weights and eight
traced33-token-context checks. Successful screening candidates still require
headline and broader semantics validation.

## Precision-locked expert and shared-MLP controls

All36 expert_sweep_sliding cases completed: BF8/BF4 gate/up × LoFi/HiFi2 ×
11/22/44 cores × K2/11/22, BF8 down, real checkpoint weights, S33/eight traced
steps. EveryBF8 case passes; everyBF4 case fails (best low-precision minPCC
~.99324). Fastest correct screening candidate is44-core K11 LoFi,4221.50us.
BF4's best4199.26us does not satisfy the unchanged.995 bar. Headline BF4 K2
also fails. summary.csv and commands.json preserve the cross-product.

boundary_commands.json records independent realS33 controls: shared BF8
packed DRAM1/2/3 readers pass at4190.76/4198.01/4209.27us; separate gate/up
with native fused GELU-multiply passes at4202.72us withDRAM1,4275.38us
interleaved. Packed interleaved4227.12us. BF4 shared gate/up fails.99166;
BF4 shared down fails.99427. Geometry/reader controls use the same physical
padding. BF8 expert prefill HiFi2 and L1 intermediates pass initial checks.

## Cumulative norm accuracy investigation

Combined path (sharedBF8/DRAM1, expertBF8/K11/grid44, active BF8 HiFi2 L1
prefill, all hidden decode norms sharded) fails only sliding4149 (.99354) and
full4105 (.98724). Neither candidate is accepted.
Eight real headline isolation controls in norm_isolation_commands.json identify
which boundary changes routing: sliding post-attention norm fails; full input
norm fails. The other sites independently pass. Unsharded controls pass both
kinds. Guarded combinations retain the original failed norm and combine only
passing sites; their full128-position reruns are in progress. No threshold,
failed position, cache semantics or advertised context is relaxed.

## Guarded norms and attention controls

Both guarded norm combinations pass real4096/128: sliding input+common,
PCC .99892058 and host4378.90us; full post+common, PCC .99954520 and
host4837.07us. Prefill PCC .99856340/.99943331. See
combined_guarded_commands.json and result pairs. These are candidates;
final defaults are not selected yet. Source now avoids retaining duplicate
BF16 expert weights when active prefill uses the same BFP8 weights as decode.

The BFP8 cache candidate was adapted beyond the initial validation error:
fill tensors cast to cache dtype, decode update tensors remain BF16, and test
validation accepts BF8 without changing orchestration. Real sliding33/8
runs deterministically but fails at34 (.99098431) and40 (.99472697).
See attention_cache8_adapted2_layer0_33.json/log. Attention controls use exact
real weights, reverse paged mapping and traced decode. QKV quantization probes
keep the broadcast projection to isolate weight precision (not BF4-storage
performance claims). Attention_commands.json records native SDPA and output
projection precision controls for both kinds. Headline and combined-policy
checks remain required for screening winners.

Projection AutoFix: qkv4149_small.json finds one-hot products exact, two-active
products already differ, and phase0-only operands have the same error at every
fidelity. The subagent proposes masked lane partitions in otherwise padded M
rows to remove intra-dot reduction error; qkv probe now includes that candidate.
No relaxed PCC, token exclusion, host fallback or substituted HF output is used.

## Projection AutoFix verified and precision frontier

Lane partitioning in otherwise padded M rows fixes the compensated projection's
intra-dot accumulation error without host execution. qkv4149_lanes.json gives
recorded-activation max error .0000258 and traced component348us (three terms),
versus broadcast .00000952 and2545us. headline_lanes_retry_commands.json tests
whole real4096/128 layers: sliding minPCC .99892503, host2198.77us; full .99954771,
host2325.06us. Two-term headline checks also pass (.99894153/.99954286) at
2142.68/2265.19us. These are passing candidates, not final defaults. The first
whole-layer attempt hit ttnn.Shape slicing at construction; explicit dimension
indexing fixed it before reruns. No cache/output semantic change.

The full36-case expert geometry sweep passes both BF8 andBF4 at S33/8.
Headline BF4 gate/up fails full attention (min .9931262), so screening alone
is insufficient. BF8 output attention weight similarly fails full headline
(.9835581). BFP4 expert DOWN independently passes both headline kinds with
lane3 QKV: .99564366/.99637312, host2189.32/2315.20us. Separate packed-expert
controls and down88-core controls are in role_commands.json. QKV packed versus
separate projections now use matched BF8 gate/BF4 down, LoFi experts and two-term
lane16 QKV while sweeping32/64 cores and K11/22; results remain candidates.

Native SDPA AutoFix identified an actual incompatible FP32 half-DST contract
in the reduction tree (five slots needed). --sdpa-full-sync fixes catastrophic
long-context failures, but headline PCC still fails sliding .9715967/full
.9871193. AUTODEBUG_sdpa.md records source and exact failing positions. BF16
Q/output boundary controls will distinguish mandatory format loss from native
intermediate arithmetic before rejecting the remaining native path.

## Provisional defaults and public contracts

The cumulative default selects two-term lane16 QKV, 8x8/K11 HiFi4 packed
projection; expert BF8 gate/BF4 down, LoFi,44-core/K11; BF8 HiFi2 active-union
prefill in32-token batches; BF8 shared DRAM1/K11; guarded sharded norms and
sharded L1 residual. default_candidate_commands.json passes both4096/128 kinds,
minimum decode .99558501/.99637605, host2135.65/2273.24us; prefill host289442/
334323us. Final profiling and any remaining candidate comparisons are pending.

pytest_default.log records4 passed; tests forbid FunctionalDecoder._forward
and compare S65 against fused output without relaxing the .995 equivalence bar.
pytest_results preserves compact result JSON. contract_commands.json passes
batch32, prefix continuation and request reuse for both kinds. All four maximum
context runs pass; long_contract_commands.json preserves exact reference files
and commands. Every TT prefill row executes; the HF oracle samples291 query
rows. No capability reduction. doc/context_contract.json records changed
persistent expert storage and the unchanged paged BF16 cache sizes.

128-token prefill initially hit L1 allocation failure (839680 bytes per bank
needed,408576 free). Releasing private projection temporaries before down and
weighting down in place fixes allocation; correctness is identical. Whole4096
sliding comparison then chooses32 tokens (289409us host), over64 (318842us)
and128 (335123us), same PCC. Shared K22 initially overlaps static CBs; readers2
and3 make it legal, but whole-layer controls are slower than K11/reader1.

## Extended stress AutoFix (in progress)

The 1025-prefill/512-traced-decode stress failed the unchanged .995 HF PCC bar for both defaults. Paired fused controls also fail, but direct saved-output comparison demonstrates additional optimized regressions; these are not waived as inherited. See `AUTODEBUG_stress.md`, `stress_pair_cpu_comparison.json`, and `stress_pair_*_layer{0,5}.json`. The paired tensors remain local and ignored.

A one-variable decoded expert-down BFP8 control (`stress_down8_commands.json`) removes new sliding failures at1108 and1175; isolated1519 persists (.974363). Full1398 remains .994985. This verifies continuous BFP4-down loss as one contributor and refutes it as the sole cause. No runtime default changed yet. Separate normalization and QKV controls follow. CPU-only BF16-cache/RoPE controls are being reconstructed on the identical seeded input stream to explain inherited discrepancies. No accuracy threshold or advertised capability has changed.

### Verified stress repair controls

`stress_verified_comparison_layer0.json`: terms3 QKV plus BFP8 decode-down removes every additional optimized HF failure; all512 paired outputs pass directPCC, minimum .998895223973. `stress_verified_comparison_layer5.json`: common-only sharded normalization with original terms2/BFP4-down likewise has no additional HF failures, minimum directPCC .996061028268. Both retain exactly the fused failure-position set on these Gaussian inputs. CPU cache/RoPE controls reproduce a subset of the shared failures; remaining shared failures are controlled by paired fused results, not claimed entirely explained by cache rounding.

Same repaired policies, 4096/128 warmed traced host median: sliding grid64 2210.621us vs grid110 2209.371us; full grid72 2303.309us vs grid110 2294.614us. Both grids pass headlinePCC. These are candidate host-wall values, not device-time telemetry. See `headline_stress_correct_geometry_commands.json` and matching JSON. Text-derived actual layer-activation fixtures and precision comparisons are still pending; no precision default selected solely from Gaussian stress.

### Watcher controls

Both stress-repaired candidates with110-core QKV pass real4096/128 optimized correctness under `TT_METAL_WATCHER=10`, separately from all profiling. No assertion-disable flags were set. Sliding minimum decodePCC .998925026174, full .996351514969; runtime audits/program-cache/deterministic replay pass. Exact commands/env are in `watcher_repaired_commands.json`; captured watcherlogs are `watcher_logs/layer{0,5}/watcher.log`. These cover the repaired candidate policy; final-policy changes will be checked as needed.

### Actual text activations control precision selection

`actual_text_fixture_manifest.json` records pinned tokenizer/text/source/config hashes and streaming HF layers0–4 to capture genuine layer0/5 inputs. This is fixture construction for single-layer tests, not full-model bringup. FP32 oracle captures are transported as explicit BF16-rounded inputs to both HF reference and TT. No test threshold changed.

Original fast defaults pass actual-text4096/128 and1025/512 for both kinds. Sliding headline minimum decodePCC .995388274209,512step .995526104127; full headline .998328713988,512step .998456537789. Thus slower BFP8-down/three-term QKV/common-only norm controls cannot be chosen solely because Gaussian stress prefers them. Gaussian discrepancies remain explicit in diagnostic reports; actual-text stress supplies model-representative acceptance evidence under the user precision requirement.

Sliding BFP4 gate/up fails actual text headline (.992109830706), while full BFP4 gate/up passes (.995955897404); broader real-input geometry/stress controls follow. `actual_text_first_commands.json` and `actual_text_second_commands.json` contain exact commands. Original optimized actual-text traced host medians: sliding2138.290us vs fused5067.964us; full2270.984us vs fused5519.467us. These are candidate host times, not final device measurements.

### Native attention and actual-input precision sweep

The native paged SDPA candidate was adapted to BF16 queries and explicit
`dst_full_sync_en=True` for FP32 destination accumulation. Both layer kinds pass
actual-text headline and extended512-step checks; BF8 paged storage also passes.
The public optimized cache validator now accepts matching BF16/BF8 cache pairs;
prefill fill values follow cache dtype, while decode update inputs remain BF16.
This supersedes Gaussian-only rejection of native SDPA/cache precision.
`actual_text_native_combined_commands.json` and `actual_native_qkv_commands.json`
(where present; exact named candidate JSONs are authoritative) retain candidate
evidence. Static integration baseline `actual_native_qkv_r0_k0_layer{0,5}.json`
records warmed traced host1374.826/1478.055us, compared with actual fused
5067.964/5519.467us. These are candidate host-wall results, not device telemetry.

DRAM-sharded lane QKV was adapted after a real static-CB overlap: align packed N
to the LCM of bank/readers and input-core geometry instead of an unnecessarily
large common multiple. Sliding reader2/K11 then runs legally at1383.544us;
reader1/K1 and reader3/K11 also pass but are slower. Full reader1/K1 and reader3/K11
pass at1638.774/1514.025us; reader2/K11 and largerK22 still exceed local buffers.
The legal adapted family loses to interleaved QKV under the same native policy.
Interrupted development NameErrors are integration failures, not kernel limits.

`actual_native_sdpa_config_commands.json` tests decode LoFi/HiFi2 and32/110-core
SDPA grids, plus prefill LoFi/HiFi2, on both real activation fixtures. All pass;
decode savings are small, prefill LoFi improves full attention by about3ms.
`actual_native_role_precision_commands.json` isolates expert/shared activation,
shared gate/down and attention QKV/output weight precision. Sliding QKV BF4 passes
headline minPCC .9950572 at1302.808us; full QKV BF4 fails .9942133 while BF8 passes
.9959558 at1417.565us. Output BF4 passes both but combined/stress evidence is
required before selection. Full expert BF4 passes real inputs; sliding gate BF4
fails even with BF8 down and multiple core/K geometries. No threshold is lowered.

Current default policy remains provisional until combined precision, contract,
watcher and same-fixture device profiles are complete. Historical Gaussian
controls remain diagnostic evidence and are not used alone to select a slower
precision policy over a real-input winner.

### Complete-layer fused profiles and integrated real-input AutoFix

`actual_fused_profile_commands.json` collects the same actual4096/128 fixtures.
Complete firmware-window prefill/decode device us: sliding2813296.579/5146.563;
full2825257.235/5586.547. Both `tracy/actual_fused_layer{0,5}` directories contain
advice-enabled report tables/CSV,128-replay timing summaries and command journals.
The native-cache estimator now uses chunk-rounded actual causal/window spans;
`roofline_native_basis.md` records its source and assumptions. No host-time value
is substituted into the device fields.

All four aggressive integrated1025/512 candidates failed real-input correctness.
`AUTOFIX_actual_precision.md` records source and paired controls. Same-policy
sliding QKV BF4/BF8 pairs isolate six BF4 misses (minimum.993737384) and all512
BF8 passes (minimum.998308950 with expert-down8). BF8 QKV/output with down4 also
passes all512 (minimum.995523980), so down8 is not selected merely for Gaussian
accuracy. Full output BF4/BF8 paired controls similarly change failed.993434071
to all512-pass.998536234. Full gate4/down4 with BF8 attention passes.996943464.
Generalized router alone fails sliding at1313/1459 while full passes; a BF16
logits control and softmax-invariant centering adaptation are being measured.
No failing headline winner is published as the final optimized path.

An actual-input CPU pilot validates streamed HF preceding layers for the full
context fixture: layer0 minimum per-token PCC.9999999999999045, layer5 after five
layers.9999999999875995 against monolithic captures, no layer0 routing changes.
Explicit sliding cache classes avoid the installed HF zero-shared-layer cache
construction's unbounded fallback. The full262144-token actual fixture and
hash-bound sampled references are being generated on CPU, with no TT device use.

### Direct QKV and precision-matched geometry closure

The native FP32-input direct QKV candidate removes lane masks, activation-term
expansion and partial-row reduction while retaining configured prefill. Actual
1025/512 controls pass with sliding HiFi4 and full LoFi. Sliding direct LoFi
fails even with generalized routing disabled, so its rejection is based on real
inputs. Centering sliding router scores before BF16 native routing restores its
512-step acceptance; full routing needs no centering. Runtime setup cleanup
shares identical phase weights and releases unused FP32 source rows and lane
masks without changing forward operations.

`actual_direct_geometry_commands.json` compares packed110-core/K11/K22 against
legal separate K11/K22 and adapted BFP8 DRAM-sharded readers2/3. Packed110/K11
wins both kinds. Reader1 buffer overlap is not the family rejection: legal
reader2/3 runs pass and are slower. `actual_direct_mlp_geometry_commands.json`
then compares matched final expert precision at44/K22,22/K22, down88, and legal
separate gate/up K11/K22; packed44/K22 wins both kinds. Full shared BFP4 also has
matched reader1/2/3, K22 and separate controls. These are warmed traced host
measurements; final selected-default device profiles remain required.

All maximum-context actual-input fixtures and hash-bound sampled HF references
are ready (`actual_text_long/verification.json`, `LONG_CONTEXT_ACTUAL_INPUT.md`).
They cover262144 and262143 for each meaningful layer kind. No Gaussian-only
precision rejection is used to lower the supported context.

### Final configuration selection

`actual_direct_output_commands.json` compares explicit64/110-core output matmuls,
HiFi4/HiFi2/LoFi, and legal DRAM reader1/2 layouts. Reader1 output is also carried
sharded through the next normalization. Every layout runs correctly; DRAM
candidates are slower, including the retained-shard variant. The winning explicit
interleaved program is now owned by `OptimizedAttention.project`, with FP32 output,
K16, legal per-core N/subblocks and no host conversion.

`actual_static_tune_commands.json` reproduces that path, tests K32, L1 output,
expert/shared HiFi2 controls and large prefill2D under the final BF8 QKV policy.
Expert/shared LoFi wins both kinds. K32 loses, L1 output improves both, full
shared reader2 improves over reader1. Prefill2D loses sliding and is within
sample spread for full, with unchanged decode; the existing prefill program
selection remains. Raw host samples are retained, not substituted for device time.

`actual_final_precision_commands.json` validates512 real-input decode positions
per candidate. On direct QKV, BFP4 QKV/output fail both kinds: sliding minima
.993831699/.993118161, full .989203513/.992120668. The BF8 controls pass.
Sliding QKV HiFi2 passes .995523605 and is faster than HiFi4 in headline and
extended controls. Full output LoFi passes .996332265 and is faster in the
compatible L1-output/readers2 path. Sliding output LoFi passes but loses in that
compatible layout, so HiFi4 remains. No synthetic PCC selects the policy.

The promoted defaults use direct packed QKV110/K11, expert44/K22, BF8 cache,
active prefill experts4/4, and explicit output64 sliding/110 full, K16/L1.
Sliding uses expert gate8/down4, shared8, centered generalized routing, all
sharded norms, QKV HiFi2 and SDPA/output HiFi4. Full uses expert/shared4,
expert activation8, raw generalized routing, post-common sharded norms,
and QKV/SDPA/output LoFi. Public contract runs now exercise these defaults.

### AutoFix: actual request-reuse precision failure

First final-policy contract pass found one full-attention failure: request6,
2049-token prefill, decode PCC.994824797. The fresh identical recorded-input
window reproduces.994824794 with deterministic tracing; stale cache/page reuse
is refuted. `reuse_control_commands.json` holds isolated controls: outputHiFi4
.994877828, shared reader1 .994824794, expert activationBF16 .994821207 all fail;
FP32 routing .998172713 and centered composite routing .998181955 pass. Only
the routing representation boundary is changed. Common-score subtraction is
softmax/top-k invariant in exact arithmetic and reduces BF16 score quantization
loss; actual expert rank swaps were not measured and are not asserted.

The selected default now centers generalized-router scores for both layer kinds.
Native BFP8 cache and expert/shared BFP4 remain. At decode position2049 the native
128-token read chunk rounds to2176tokens, fitting both fresh3072 and reuse4096
allocations. `AUTODEBUG_reuse.md`/`AUTOFIX_reuse.md` record source diagnosis and
controls. New `validated_*` artifacts recheck final defaults; earlier `final_*`
artifacts are the pre-repair control, not final signoff.

### AutoFix: maximum-context full-prefill failure

`validated_long_262144_layer5.json` fails actual sampled prefill PCC.991703279;
all three end-context traced decode checks pass (.997745/.999088). The advertised
context remains262144. `actual_long_control_commands.json` raises only prefill
down, only gate/up, or both toBFP8: .992009353/.992018342/.992332985, all failures.
These controls refute expert precision as a sufficient repair; no such change is
kept. Fresh source diagnosis focuses on chunked full-prefill attention, whose
later chunks consume paged BFP8 K/V and whose legacy kernel repeatedly packs its
running output numerator in BF16. End-context decode uses a different kernel.

`actual_long_attention_commands.json` starts independent attention controls.
Kchunk256 with unchanged query config exceeds static L1 (2,012,160>1,572,864B)
in the initial non-chunked BF16-K/V attention call. This is an unadapted attempt,
not a family rejection; smaller query blocks are the next layout-compatible
adaptation. HiFi2 with the original chunks is a separate same-cache control.
Per-sampled-row PCC diagnostics were added outside model execution to expose
whether errors grow with position; the unchanged aggregate.995 gate remains.

The same-cache HiFi2 attention control passes maximum-context aggregate
PCC.998700577 with unchanged BFP4 experts. Adapted paged-onlyQ64/K256 avoids the
initial BF16-K/V allocation problem: LoFi runs legally but fails.994555053;
HiFi2 at identical geometry passes.999050538, with all291 sampled rows≥.995.
This proves sensitivity to the prefill SDPA compute boundary, not a specific
kernel defect or expert-precision limitation. `AUTODEBUG_long_prefill.md` records
source, allocation math, and exact controls.

Diagnostics exposed final-query LoFi PCC.9592894097 hidden by the aggregate.
The long-context test now adds a targeted regression: the final two prefill
queries must satisfy the existing.995 threshold independently, matching the
positions already checked for decode. The aggregate gate is retained and its
original status is recorded separately. This is additional regression coverage,
not a retroactive claim about earlier tests; other per-row PCCs remain diagnostic.
HiFi2K128 and HiFi2Q64/K256 both satisfy this endpoint regression. A final
paged-onlyQ32/K512 LoFi control checks whether larger key blocks suffice.

Expert gate/up K blocks44/88 are now independently configurable from downK22.
`expert_gate_block.md` records source-backed CB bounds and the optional patch;
this prevents an invalid down-K divisor from hiding a legal dominant-gate
candidate. The larger gate-only blocks receive the same real4096/128 correctness
and traced-timing comparison as existing candidates. No default change is made
without that result. Repository pre-commit Python checks pass after formatting.

## Final-context repairs and last topology comparison

Maximum actual-text full prefill exposed a LoFi attention precision failure. Isolated expert gate/down BFP8 controls failed to repair it. HiFi2 paged attention repairs it. See `AUTODEBUG_long_prefill.md` and `long_prefill_config_results.md` for exact controls and L1 adaptation: Q64/K256 is changed only for paged BFP8-cache calls, retaining the first BF16-K/V call's legal geometry. New end-query regression rejects Q32/K512 LoFi even though its aggregate PCC passes. Selected Q64/K256 HiFi2 passes all291 sampled rows at262144; no context restriction was added.

`expert_gate_geometry_results.md` closes independent sparse gate K22/44/88 controls while keeping down K22, including matched 22-core and separate-gate controls. Packed44-core gate K44 wins both kinds. Native SDPA now retains BF16 through head concat and projection input; FP32 projection output remains. `native_sdpa_boundary_results.json` shows identical PCC and a full-layer traced host improvement, with sliding within timing variation.

`stress_selected/summary.json` records the cumulative selected-default1025/512 run before the router-matmul experiment: both kinds pass all HF checks and direct fused comparisons, with deterministic traces and no runtime host fallback. Sliding minimum HF decode PCC .9955236049581246; full .9963231548432059.

The final topology audit identified the remaining broadcast multiply/reduce router-score projection. `probe_optimized_router_direct.py` tests native FP32-input matrix multiplication, preserves centering/native gate/prefill, and records actual tensor descriptors. Initial headline sliding HiFi4 passes with unchanged PCC and reduces traced host decode from about1071 to988us. Block/fidelity sweeps and extended/request-reuse controls are in progress; no final router policy or stage pass is claimed yet.

## Integrated final router and validation snapshot

Applied native router with automatic per-kind policy: sliding4-core/K22/HiFi4, full4-core/K44/LoFi. SlidingHiFi2 failed one actual512-step position; HiFi4 repairs it. FullLoFi passed512steps and the isolated reuse window. Runtime SHA256 `e81018299b722aa81eae0e4e9ec3ec638520adcb3bdd85d3c2c8b429dfa0e370`; repository pre-commit passes without changes. `verified_contract_commands.json` is the current final-default journal; source remains immutable during that batch. Full B32/prefix/reuse/BF16cache/max262144/max262143/headline/watcher and sliding short contracts have passed at this snapshot. Remaining batch checks and final profiles are still running/pending.

## Final correctness gates complete

`verified_contract_commands.json` contains18 successful commands on runtime`e81018299b722aa81eae0e4e9ec3ec638520adcb3bdd85d3c2c8b429dfa0e370`:16 per-kind contract/headline/watcher commands,4-case pytest, and paired real1025/512 stress. `verified_validation_summary.json`, `verified_watcher_summary.json`, and `stress_verified_defaults/summary.json` index results. Both exact HF and fused-preservation stress gates pass with no shared/excluded HF failures; minima.995523605 sliding/.996162947 full. All public contracts forbid functional fallback. Watcher interval10, disabledfeaturesNone, separate from profiling. Whole-layer traced host headline medians987.932us sliding/1076.212us full; warmed prefill223737.166us/195215.563us. Device timings remain pending final profiles; host values are not telemetry device values.

## Post-profile work: sampled sliding-prefill discrepancy and compact experts

Final sampled diagnostics show four sliding-prefill rows below.995 at262144, despite passing aggregate/end-tail/decode gates. The exact real-input fused control `actual_fused_long_sliding.json` passes all291 sampled rows. `actual_long_sliding_hifi2.json` does not repair this: five sampled misses remain, with position140287 .981222 versus currentLoFi .982346. No HiFi2 change was retained. A fresh xhigh source diagnosis and isolated precision controls continue under AutoFix; the stage is not complete.

Current final-profile snapshots are complete for both kinds in `tracy/actual_optimized_layer{0,5}`. They measure223968.078/195086.517us prefill and1069.683/1147.548us mean complete-layer decode device windows. Compared to the same-workload fused profiles, decode improves4.811x/4.868x. These are current-runtime measurements, not stage signoff. Same-run host/profile windows and separately measured unprofiled987.932/1076.212us are kept distinct in reconciliation JSONs.

Advice identified router1x1 output subblocks; matched2-core/1-core controls are running. Source audit also identified existing indexed sparse matmul with compact8-expert outputs, plus an existing weighted-reduction composite that may remove dense128-expert activation and transpose work. These material existing-op candidates are being adapted and tested now, not deferred.

## Sliding-prefill precision repair localized

`AUTODEBUG_sliding_prefill.md` and `AUTOFIX_sliding_prefill.md` record the fresh
source diagnosis and the parent's isolated actual-input controls. Only prefill
gate/up BFP8 repairs all 291 sampled maximum-context sliding rows; down BFP8,
QKV BF16, output BF16, and SDPA HiFi2 each leave misses. The passing gate/up
control retains BFP4 down, LoFi attention, BFP8 cache, and exactly identical
decode results. Minimum row PCC is .995109737003 at position 32; position
140287 improves to .997184060335. No router-ID flip is claimed.

The parent applied `sliding_prefill_gate8.patch`: automatic prefill gate/up is
BFP8 sliding/BFP4 full, and recorded-text long-context checks now require every
sampled row to pass .995. No-fixture gate behavior is unchanged. Applied runtime
hash is `0aabcac2109a35b436c78ca6322ba4e88331abdab39e8271e14ea5235af9938a`.
Static gate replay rejects the failing controls and accepts gate/up BFP8 plus
the final full-attention and fused controls. Default-path hardware reruns and
final performance remain pending. Setup aliases the existing sliding BFP8
decode gate/up, removing the separate BFP4 gate/up copy; its static payload is
285474816 bytes, with final device allocation accounting still required.

## Applied indexed experts and RoPE producer layout

`compact_expert_commands.json` and `compact_geometry_commands.json` prove indexed compact8-expert execution at unchanged weights/fidelity. Smaller22-core, larger88-core down, gateK22/K88 and tuned separate paths lose;44-core gateK44/downK22 remains. Accurate fused GELU becomes slightly faster with compact8 outputs and preserves every measured PCC, unlike its earlier expanded128-slot trial. Weighted reduction loses both expanded and compact comparisons. `late_topology_results.json` keeps exact measured host samples.

`rope_layout_commands.json` compares TILE and ROW_MAJOR2D decode table uploads with identical HF values. Both kinds preserve every headlinePCC; row-major saves30.367us sliding/54.776us full. Applied public `decode_rope_layout` capability and setup-only harness consumption;4D prefill tables stayTILE. OldTILE inputs remain accepted; no caller tensor content cache was introduced.

The applied combined runtime is`b513a1b40988b33a359acb7d7809696eebf81a7197756e3ec3f943692182ab83`, including slidingprefgate8, compactexpertindices, fusedaccurateGELU, RMpreferreddecodeRoPE. `summarize_perf.py` now derives indexed expert activecount from native use_indices plus compactoutputshape, with optionalindexoperandcross-check. The installed reporttool ignoresuse_indices; finaldecode-only reports will use explicit`--active-experts 8` after verifying rawrows. Prefillreports retainvariableunioncounts and neveruse thatoverride. Finalruntimevalidation/profiles remain pending while explicitprefoutputprogram/fidelitycontrols run.

## Selected cumulative runtime for final validation

Runtime SHA256 `3d51014f98128dfb21bb484fcece50993524b967754f68f6dfe825ae7f472ba9` integrates prefill output grid11x8/K16/LoFi with DRAM input/output and full input norm sharding. `prefill_output_results.md/json` compares 28 completed real-input commands; inputL1 adds movement without a meaningful sliding gain and slows full. `full_all_norm_commands.json` proves full4096/128 and1025/512 pass, with headline traced host832.355us versus864.212us on the previous post/common-only path.

The native sliding headnorm control completed but fails real512-step positions1257 and1459 at.994804/.994419. Its faster740.762us headline cannot pass the unchanged.995 correctness gate; precise head norms remain. This is an actual-input rejection after a legal completed run, not a synthetic or API-error veto. `native_headnorm_control.md` records source-level arithmetic differences without claiming measured router-ID flips.

Repository Python pre-commit passes in `precommit_v4.log`. `/tmp/gemma_validate_v4.py` runs16 public-contract commands plus4-case pytest and paired actual1025/512 stress serially; exact arguments/returns are persisted in `validated_v4_contract_commands.json`. The runtime stays frozen during these checks. Both maximum and nonaligned near-maximum actual-text prefill now gate all291 sampled rows independently; historical aggregate-only acceptance is not relabeled. Final profiles/review/commits remain pending.

## Final cumulative correctness passed

`validated_v4_validation_summary.json` independently rechecks18 successful commands and16 matching-runtime reports, four pytest cases, exact-HF/direct-fused512-step preservation and Watcher lifecycle logs. Both kinds pass all291 sampled maximum/near-maximum rows; minima .995099173 sliding/.996014477 full. Stress HF minima .995525862/.996240662 and direct-fused minima .995146484/.996320299 pass without exclusions. `validated_v4_tile_rope_commands.json` adds passing4096/128 TILE-input compatibility for both kinds; selected ROW_MAJOR setup remains faster. The original source and test evidence are unchanged during validation. Final native profiles and matched operator controls are being collected separately from Watcher.

## Resumed attempt a5411fee: installed skill additions

The resumed user selected installed optimize0.1.14, including explicit short/tail prefill timing, minimal_matmul comparisons and OPT015 alternating isolated DRAM-reader checks. Existing runtime3d51014f and v4 correctness/headline profiles remain the current validated baseline. Environment package exports and enabled tt-autodebug inventory are present; bounded `tt-smi -ls --local` discovers all four BlackholeASICs on twoP300c boards. No reset was needed. Resume drivers are now persisted here rather than only in/tmp.

The earlier operator audit had a Tracy CLI quoting failure: `--default-overrides` JSON was split, argparse rejected it, but Tracy returned0 without a runner JSON or native CSV. This was never device correctness/performance evidence. `tracy/operator_failed_cli_layer0` and `operator_audit_failed_cli_commands.json` preserve it. The named-case wrapper `tests/profile_optimized_expert_case.py` removes JSON from Tracy's shell boundary; `run_operator_audits.py` requires a passing runner and native signposts. All six native operator audits then completed.

`final_perf_advice.md/json` now bind actual compact8 rows, both current profiles and all six candidate profiles. `final_roofline_audit.md/json` reconstruct whole-layer timing, useful FLOPs, indexed weight bytes and KV-read spans from native rows. Layer0 same-run reconciliation was regenerated against the completed two-command profile journal; numerical measurements were unchanged. The new attempt's local telemetry packet is copied only from the existing matching v4 measurements, with stage completion still pending.

`probe_optimized_minimal_prefill.py` compares existing TTNN minimal_matmul with unchanged real projection inputs, weight/output dtype, fidelity and memory boundaries. Output candidates retain every headlinePCC. QKVK16 passesheadline and512steps for both kinds; slidingK8 also passes512, while fullK8 failsheadline minPCC .994989324 and is rejected. Selected QKV candidates still require the strict maximum-context gate. Same-process alternating whole-prefill output-backend timing resolves small gains before choosing a backend. No runtime change has yet been made for these candidates.

## Integrated minimal prefill selection

Both selected minimal-QKV configurations pass the262144-token gate with all291 sampled rows. Output projection same-process eight-pair measurements reject sliding minimalK16 (221729.583us versus221598.143us regular) and select full minimalK8 (189143.102us versus189347.941us regular). These are synchronized warmed host whole-prefill times, not device latency. Runtime169c0d97d7d0e9f35d97633f133305f1088b987ef625693d3faa100d25c3e67b integrates QKVK8 sliding/K16 full and minimal output only for full attention. Decode implementations are retained. The persistent run_validated_v5.py repeats affected public contracts, strict long rows, headline accuracy/cache checks, separate Watcher, small-tail pytest and512-step direct-fused preservation before accepting this new default.

## V5 acceptance, reader controls and review repairs

The integrated v5 runtime169c0d97 passes all18 campaign commands, including both strict maximum/near-maximum contexts, all291 selected rows, B32, prefix/cache contracts,33/65 pytest and real1025/512 stress. Separate native4096/128 profiles measure221240.953/187940.530us warmed prefill and881.434/900.013us whole-layer traced decode. `validated_v5_validation_summary.json` and `actual_optimized_v5_profile_commands.json` preserve exact attribution.

`prefill_boundary_commands.json` completes16 matched fused/optimized warmed cases at65,1023,1024,1025. `prefill_boundary_summary.md` records physical chunks and all correctness/program-cache guards. `dram_reader_results.md` completes alternating isolated readers1/2/3 with exact production dtypes, fidelity, captured inputs and common padding. The two QKVK11/reader1 cases exceed per-core L1 by exact allocation calculation; all28 legal cases pass. All22 current-source whole-layer controls in `reader_layer_v6/commands.json` pass. No material layer gain supports changing direct QKV/output or shared reader1/1 sliding and2/2 full. The full down-reader1 apparent0.047us gain lies within overlapping timing spread. The reader probe formatting incident and initial wrapper Shape-slicing error are preserved with phase hashes and retries, not erased.

Independent review exposed retained allocations under live traces and a tight-cache page-table read risk. Fresh AutoDebug plus isolated tracker runs distinguish observed allocations from unproven address overlap. Runtimeb585a21f removes unused final sliding-tail clones and selects Q64/K128 only when full-prefillK256 would exceed logical cache capacity. The request-reuse test warms exact signatures before capture and forbids cache misses afterward; no tracker exclusions are used. Both nine-request full-tracker runs, eight tight-capacity cases, current4096/128 headlines, separate Watcher and four real-tail pytest cases pass. `AUTOFIX_review_contracts.md`, `source_delta_v6.md` and `validated_v6_validation_summary.json` record current/inherited coverage explicitly; old reports retain original source hashes.

`run_v6_remaining.py` completed tight-cache and correctness campaigns, then stopped before hardware profiling because its validation manifest was not yet present. `v6_final_campaign.json` preserves that metadata failure. The completed manifest now allows a standalone profile retry; there was no device/profiler failure. Before the retry, native advice audit identified untried minimal-QKV HiFi2 and11x10 controls. These are being measured before final policy/profile freeze. No lower-performance policy or untried material advice is accepted solely to close the stage.

## Final advice audit: measured minimal and placement controls

All six `minimal_advice_v6` headline controls pass, but separate-process spreads overlap. `minimal_pairs_v6` therefore repeats eight alternating baseline/candidate pairs with both signatures warmed and program-cache misses forbidden. HiFi2 QKV improves median paired whole-prefill host time473.456us sliding (7/8 pairs) and545.573us full (8/8). Grid110 gives no resolved gain: +4.3035us sliding (4/8) and+87.4815us full (2/8), so11x8 remains. `minimal_pairs_v6_summary.md/json` preserves all samples and same-process setup/compute metadata. The candidate’s output/cache is restored before the original128-position correctness gate. Lower fidelity is undergoing independent1025/512 and262144/all291-row acceptance before adoption; no change is yet claimed in the runtime.

`prefill_placement_v6` completes all three legal L1-input controls under locked native programs and precision. Whole-prefill medians221296.830us sliding shared,188075.827us full shared and188397.882us full minimal output show no measured improvement over matched screen baselines221091.133/187705.330us. The report’s shared88→110 advice requires work redistribution, because its original configured grid is already11x10. A legal transpose-multicast110-workblock program is being measured rather than treating configured grid size as rejection evidence.

`package_evidence.py` preserves exact original measurement/report bytes and historical executed scripts in deterministic gzip companions where repository size or formatting hooks would otherwise rewrite them. `evidence_archives.json` records original/archive SHA256, sizes and reconstruction. Local originals remain intact. The reviewer independently verified all77 initial archives. Raw tensors/Tracy data and pytest scratch symlinks remain ignored; primary result summaries are plain files.

## Adopted full-only prefill HiFi2; final v7 validation

Sliding HiFi2 fails the real262144-token strict gate at sampled position32:0.9949930133358956 versus HiFi4 control0.9951303778312409; aggregate and512-step checks pass, but do not override the row gate. Sliding therefore keeps HiFi4. Full HiFi2 passes all291 sampled rows (minimum0.996029643684647), boundary decode and512 steps. The parent acceptance driver encountered a post-run metadata error treating the long decode list as a dict; `minimal_hifi2_acceptance_v6/metadata_failure.json` records it, and `full_long_resume_command.json` records the independent successful continuation. No failed report was overwritten or numerical bar relaxed.

Both legal redistributed110-core shared controls complete and lose their screen comparison (221857.644/188432.310us versus221091.133/187705.330us). All remaining prefill placement/grid advice is now measured. Runtime`daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b` adds only a canonical prefill-QKV fidelity override: auto resolves HiFi4 sliding/HiFi2 full, copying all other compute flags. Weights, blocks, output/cache dtypes, decoder trace policy and v6 lifecycle/bounds repairs are unchanged; the complete b585 source snapshot is retained.

`run_validated_v7.py` exercises12 commands: all8 full public contracts (including both maximum/nonaligned contexts and full-tracker reuse), sliding headline/Watcher, four-case real-tail pytest and both-kind direct-fused/HF512-step stress. Completed early gates include all full public contracts, both current4096/128 headlines and Watcher runs. `run_v7_boundaries.py` additionally refreshes the tight1025/cache1152 K128 native bounds case and full65/1023/1024/1025 warmed prefill. Final current native profiles and review follow those gates; old device timings are never relabeled with the v7 hash.

## Final native profile results and last placement adaptation

Both v7 Tracy runs and all eight advice-enabled report commands pass. Complete device prefill is221262.879259us sliding/187438.157037us full; mean128-position traced decode is881.242928us/899.724676us. `actual_optimized_v7_profile_commands.json`, per-kind native projection audits and same-run reconciliations bind the current source. The independent CPU roofline audit reconstructs every whole-layer window and passes without corrections; summed kernels are not used as roofline denominators. The separate unprofiled default medians are221110.611/187654.130us prefill and796.2788/833.379133us traced decode.

After full QKV changed to HiFi2, its native report emitted inputL1 advice. The exact selected M4/K16/N8 L1 boundary trial fails before timing: static circular buffers end at1307648 while an L1 allocation begins1114112. `prefill_qkv_l1_v7_command.json` records rc1 and the failed report. This first failure does not reject the family. A smaller-M, unchanged-K/fidelity/dtype adaptation is being measured against the selected DRAM/M4 whole-prefill path.

The legal M2/K16/N8 L1 adaptation passes all headline checks and wins8/8 pairs (median paired−809.1405us versus DRAM/M4). M4/K16/N4 also fits and passes but has a smaller8/8 gain,−384.858us. A matched M2-only placement comparison confirms the benefit is not just smaller M: L1 wins8/8 at−664.9255us versus DRAM/M2. All use the exact selected HiFi2/FP32/BFP8 boundaries and preserve K accumulation order. The original M4 overlap is therefore an adapted capacity constraint, not a blanket L1 rejection.

The reviewer identified an avoidable extra copy in the trial: the existing final weighted input-normalization multiply can write directly to L1. A full-prefill-input-only producer-layout control is next, with M2/K16/N8 fixed. Runtime remainsdaa82 while comparing these options; current profiles are not relabeled as the eventual integration.

## v8 full-prefill producer placement

The exact M2/K16/N8 direct-producer candidate won 22/32 alternating warmed whole-layer prefill pairs against the legal copied-L1 candidate (median paired delta −128.295 µs). Both preserve headline real-weight accuracy. The separate M2 L1-placement control won 8/8 pairs against identical M2 DRAM input (−664.9255 µs); the matched 110-core candidate won only 4/8 (+2.8355 µs), retaining 88 cores. `qkv_l1_results.json` preserves all seven controls including the initial M4 circular-buffer overlap and the legal M2/N4 retries.

Integrated runtime SHA256 `5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898` produces full-prefill input-normalization output directly in L1 and selects M2; sliding prefill and all decode policies are unchanged. `run_validated_v8.py` and `run_v8_boundaries.py` bind affected correctness, capacity, trace-lifecycle, Watcher, stress and tail evidence to these exact bytes. Native v8 profiling and the phase-specific prefill-router advice controls remain open at this entry.

V8 acceptance completed: all12 primary and5 boundary commands returned0. `validated_v8_validation_summary.json` reports no pending/error entries. Full-attention live-trace program count is379 and remains379 across all nine requests. Both512-position stress comparisons retain all HF and direct-fused rows. `prefill_boundary_v8_summary.json` records actual short/tail host timings; these are not device measurements.

Additional explicit pinned-Black sweep:84/85 source files pass; `tests/probe_optimized_prefill_pairs.py` retains its historical hash-bound bytes. Repository Black excludes autoports, so this optional formatting discrepancy does not bypass a required hook. Exact optional-check output is `black_final_optional_sweep.log`; final required pre-commit remains separate. No C++/CMake changed, so no build is required.

The subsequent required-isort audit found the same historical helper needed import formatting. Resolved by retaining its exact old bytes in `probe_optimized_prefill_pairs_before_format.py.txt` (SHA25613f71dfb87a9cced61bd974effdb666dee972bade6591954fcb86a8c76a09376), then applying pinned isort/Black to the live helper (b682c87a62bab2d6e66a25a6893e75d972847f7048d4831a2e0285ced72bf630). `prefill_pair_helper_format_proof.json` verifies identical import statements and identical non-import AST. Historical controls resolve their original hashes against the snapshot; new router controls bind the formatted helper. This supersedes the proposed formatting exception above.

Final v8 native profiles completed successfully. Whole-layer device prefill/decode means are221367.854917/881.410845µs sliding and186562.835556/899.335532µs full. `audit_final_roofline_v8.py` independently reconstructed both native windows, all128 decode positions and current selected configurations; it passes. The reviewer identified that minimal QKV's unspecified output memory inherits its new L1 input placement: native full QKV is L1→L1, not L1→DRAM. Correcting policy/memory/advice prose to that measured fact; existing full-capacity tests already exercise this actual layout. No measurements or historical source hashes are changed.

Final prefill-router advice closure: four actual4096/128 controls each passed32 alternating pairs, HF gates, repeat equality and stable program counts. Sliding HiFi2 won20/32 pairs (median−67.093µs), sliding L1 producer16/32 (+3.8725µs), full HiFi2 10/32 (+111.637µs), full L1 producer20/32 (−32.762µs). Every paired IQR crosses zero; no resolved whole-layer gain supports changing the HiFi4/DRAM default. Exact controls: `prefill_router_v8/commands.json` and its four reports.

Repository-required `pre-commit run --files <1479 stage-owned non-ignored files>` passed with no mutations (`precommit_final_all.log/json`). Additional explicit pinned Black now passes all85 Python files (`black_final_all.log/json`); the historical helper discrepancy is resolved with its byte-exact snapshot and AST proof.

Final staged scope check: all1546 files belong to this model's optimized decoder, tests or docs; no tensor payloads, raw Tracy traces or raw ops.csv are staged. Full required pre-commit passes without mutation (`precommit_final_staged.log/json`). `git diff --cached --check -- . ':!*.patch'` passes; historical patch context blanks are intentionally retained, consistent with the repository trailing-whitespace hook's patch exclusion. Compact perf-report CSVs and native replay tables are included as exact gzip archives because root `.gitignore` excludes CSVs.

## Independent stage acceptance

The fresh xhigh `$stage-review` reviewer returned **clean-pass**, recorded in `STAGE_REVIEW.md`, with no Required Work. It independently checked runtime5ff391, all249 archives including58 CSVs and104 logs,37 validation hash bindings, current native timings/configurations, all38 optimization checklist entries and final staged pre-commit evidence. All14 anomaly groups are classified/resolved. No runtime or hardware follow-up remains. The following local checkpoint step records the reviewed stage-owned source/tests/docs; nothing is pushed.

## Local checkpoint record

Repository `/workspace/tt-metal`, branch `gemma-4-26b-a4b-it`: reviewed implementation/tests/docs checkpoint **`a9259624f2ad89a17fcf94f7351e04c82d7faa35`**. The commit's required hooks all passed; the tree was clean immediately afterward. This documentation checkpoint records that SHA and closes the checklist. Its own SHA is recorded externally in the final required evidence packet at `bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/a5411fee-0441-4640-a0ff-c23fc77b76a6.json`, avoiding a self-referential commit hash.

The repository had no configured author identity, so commits use the same explicit per-command agent identity as the preceding stage (`Codex <codex@openai.com>`); no global git configuration changed. Both checkpoints are local only. Nothing was pushed. Stage03 requirements are complete, with independent `clean-pass`, preserved262144 context/non-aligned lengths, current device measurements, completed advice controls and no deferred decoder optimization.
