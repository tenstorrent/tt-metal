# Optimized full-model checklist — evidence snapshot, not acceptance

Current evidence on 2026-11-16: final all40 correctness, mixed/fixed-slot
replay, serving commands, sampler controls, clean performance and short/chunk
boundary TTFT, reduced native decode attribution and bounded prefill kernel
attribution and long-context smoke/census have finished. Independent round1 review returns clean-pass (stage_review_round1.md). The cumulative selected path reaches
46.6922 tokens/s/user and passes the full-model useful-byte floor. The earlier
candidate1 floor failure is retained; it is not the selected result.

Paths are relative to this directory. `Inherited` means evidence for the exact
prior source/workload, not a new final-source gate. `Pending` is required work.
`Untried` means an optimization opportunity, eligible for a Stop Rule handoff
only after that rule is explicitly recorded; it never waives correctness.
No vLLM integration or broad full-model datatype frontier is in scope.

## Actual starting point and current patch scope

- [before.json](before.json): untracked all40 B1 128-in/128-out, TTFT
  2787.782 ms; queued token-out 25.081181 ms (39.8705 t/s/u); caller-observed
  delivery 25.158932 ms; fixed-token model trace 24.601166 ms. Fixed-token
  model-only replay is not teacher forcing and neither is token-out.
- [before_prefill.json](before_prefill.json): public TTFT at logical lengths
  64/255/256/257/1025/2049, respectively 1474.063/5905.173/5350.840/5975.767/
  21825.866/43157.731 ms. Each row preserves trace IDs/program cache and device
  position advance through two resident feedback steps. This is a repeated
  AIME-token capacity/performance fixture, not qualitative evidence.
- [topology_audit.md](topology_audit.md) records shapes, bytes, rank ownership,
  next consumers, collective alternatives and the inherited rejection ledger.
  The historical layer estimate is
  `30*0.719504474 + 10*0.340836807 = 24.99350229 ms`.
  Against the stage7 starting queued token-out the difference was0.087679ms (0.351%). These
  separate workload regimes are not an additive profile decomposition. The
  separate useful-byte FUNCTIONAL ceiling is 21.631299 ms (46.2293 t/s/u),
  so a small wrapper gap does not establish the required performance pass.
- Candidate1 selected the reader3/block4 LM head, semantic prefill config keys,
  structural movement reductions, narrow32 router scan, no-clear cached K512
  input/output scans, exact direct recurrent sum and fused product+sum.
  `candidate1_performance.json` completed all40 execution with TTFT2768.434ms,
  queued22.272769ms (44.8979t/s/u), actual99-step teacher21.788870ms,
  and zero resident host refresh/readback/recapture. It **fails** the separate
  21.631299ms useful-byte floor by0.641471ms. `candidate1_teacher_accuracy.json`
  passes100 predictions at top1=98%,top5=100%,top100=100% on that source.
- `production_functional_l0.json` and `production_functional_l3.json` record
  clean exact whole-layer controls:0.719497745→0.632541260ms and
  0.341083533→0.320842714ms. Candidate1 stack estimate is22.184664941ms
  (`candidate1_layer_bound.json`), not additive device decomposition.
- The further state-slab recurrence has exact component and integrated evidence
  below and is wired through the selected full-model decoder. Final frozen-source
  all40 performance, accuracy, serving and qualitative gates pass. Current
  queued-layer estimate20.668310ms comes from final/layer_stack_budget.json;
  final token-out21.416868ms is3.62% above it before terminal allowance. The
  separate component30us gain is not an additive all40 decomposition. Context
  qualification passes in final/prefill_context32769.json.

## Completed final correctness evidence (bounded scope)

- [Prefill](final/prefill_accuracy.json) and [teacher forcing](final/teacher_accuracy.json)
  each pass100 AIME24 chat-template predictions: top1=98%,top5=100%,top100=100%.
  Their provenance records all40 layers,1×4 mesh, tracking1/skip0. Teacher timing
  in this tracked accuracy run is diagnostic, not the clean headline result.
- [B32](final/b32.json) runs all40 with32 fixed slots/context1024, uniform4-token
  prompts, reordered continuation and inactive slots1–31 unchanged. The state
  byte snapshots cover **mesh chip0**, not an all-rank numerical claim.
  Program-cache growth is0;2 resident model/sampling replays cause no token,
  position or page refresh. One activity update and2 token readbacks are explicit
  caller observations. All40 mixed replay and serving now pass in final/replay_all40.json and final/serving_commands_all40_v3.json.
- [Six-case verdict](final/qualitative/verdict.json) and
  [paired reading](final/qualitative/paired_review.md) record bounded qualitative
  PASS:128 greedy tokens per canonical chat prompt, all40, pinned matched HF
  controls. All six fresh TT texts exactly match the completed-model TT controls.
  Both TT/HF remain in reasoning at the cap; this is no completed-task score or
  proof of HF token equality. The q03 doubled phrase is shared with HF and prior
  TT. [Degenerate check](final/qualitative/degenerate_output_check.json) exits0
  with no findings. No degraded-content case was selected for a10-seed test.
- [AIME autoregressive verdict](final/aime_autoregressive/verdict.json) records
  bounded qualitative PASS for the exact161 saved prompt IDs and100 generated
  tokens. Both outputs are relevant coherent reasoning; token sequences differ
  from the first token and neither solves the task within this cap. This does
  not substitute for the separate top-k accuracy gates.
- [Tracker positive control](final/positive_control.json) actually rejects the
  deliberately unsafe allocation. Its layers0/3 scope is diagnostic only.

These results do not declare stage acceptance or satisfy Stop Rule1. Native
profile/advice and bounded prefill attribution are complete at the scopes below;
long-context smoke/census passes; independent review returns clean-pass. Source changes invalidate only the affected
final gates; retain their recorded source identities.

## Completed native and prefill attribution (bounded scopes)

[Final native audit](final_profile_audit.md) and
[component_summary.json](final/profile_reduced/component_summary.json) validate
real layers0/3 plus complete terminal/sampler at B1/context1024. All fifteen
recorded source hashes match clean all40 timing and current source. Native
rank2 work is1.266784ms model plus0.436511ms sampling; the25.63% reduced
sampling share is not the all40 share. The clean all40 sampler control is2.28%
of21.416868ms token-out. Selected head reader3/block4, cached K512 scans,
narrow32 router, FP32 state slab, active LoFi experts, BF16/FP32 direct-AR
scratch and candidate-only sampling gathers all have actual native evidence.
No generic TopK/argmax/full-vocab gather appears in the selected decode window.
The eager prefill expert router does use generic TopK; do not extend the decode
statement to all setup/prefill operations.

All advice strings have explicit dispositions in that audit, including inherited
sparse-L1/subblock evidence and the measured isolated FP32-router L1/subblock
controls. Fidelity recommendations do not authorize changing the selected
policy. Raw tables/CSV/provenance remain at final/profile_reduced; compact
final_profile_attribution.csv and three op-group tables preserve bounded
per-rank evidence. Cached embedding-AG height metadata cannot establish dynamic
row count; its producer/source establishes singleton decode. The optional WASM
copy warning does not invalidate the retained, successfully converted capture.

[Prefill analysis](prefill_performance_analysis.md) and
[prefill_role_chunk_summary.csv](prefill_role_chunk_summary.csv) separate clean
all40 boundary TTFT from the instrumented128-token reduced window. The final
boundary samples are−0.030% to+3.428% versus baseline, one sample/length: no
statistical regression or improvement claim and no matched-kernel causal claim.
The native window has12468 eager and124 traced rows, with73.964–74.084ms eager
and0.428–0.442ms traced kernel work per rank. Its15761.827ms host TTFT includes
profiler drains and is not headline TTFT. Only668 eager rows have entries in
the decode-oriented metadata inventory; inline roles/fidelity remain bounded.
Shape/source evidence identifies two64-token embedding pieces and one128-token
SDPA chunk. The largest physical-Y64 custom group (~49.435ms/rank) matches the
chunk64 recurrence family by source/shape;128 calls in the physical-Y1024
group are per-token K1024 scans, not1024-token prompt chunks. Individual inline
roles remain unresolved in the CSV. Setup-only SDPA buckets32..256 are prepared
variants, not measured logical request lengths. No final kernel-cost table for
every nonaligned logical chunk or matched before/after prefill kernel capture
is claimed; finer prefill attribution/geometry remains an untried handoff.

## Terminal geometry evidence

Every passing row below uses real reduced layers0/3 terminal hidden, unchanged
BFP8 weights, HiFi2/FP32 accumulation and exact full logits/top-1 checks. Times
are complete traced terminal component measurements, including final norm and
layout movement. They are not all40 decode improvements. Local vocabulary
width remains 62080; storage padding does not change rank offsets or expose
new logical IDs. Core count is16 and per-core output width is32 tiles.

| Readers / K-block / splits | Baseline → candidate ms | Evidence / decision |
|---|---:|---|
| 3 /4 /1 | 0.452275 →0.353635 | [head_r3_b4_c16.json](head_r3_b4_c16.json); selected component geometry |
| 3 /8 /1 | 0.451994 →0.367281 | [head_r3_b8_c16.json](head_r3_b8_c16.json); slower |
| 3 /16 /1 | 0.452159 →0.382754 | [head_r3_b16_c16.json](head_r3_b16_c16.json); slower |
| 2 /8 /1 | 0.452421 →0.371851 | [head_r2_b8_c16.json](head_r2_b8_c16.json); slower |
| 2 /16 /1 | 0.452570 →0.417201 | [head_r2_b16_c16.json](head_r2_b16_c16.json); slower |
| 3 /32 /1 | No valid timing | [head_r3_b32_c16.json](head_r3_b32_c16.json); static L1 2080704 >1572864 bytes |
| 1 /1 /1 | No valid timing | [head_r1_b1_c16.json](head_r1_b1_c16.json); static L1 2408960 >1572864 bytes |
| 1 /1 /4, ballast | 0.460045 →2.398756 | [head_pressure_r1_b1_s4.json](head_pressure_r1_b1_s4.json); legal adapted one-reader control, rejected slower |
| 3 /4 /1, ballast | 0.460104 →0.361358 | [head_pressure_r3_b4_s1.json](head_pressure_r3_b4_s1.json); passes resident L1 proxy |
| 3 /2 /1, ballast | 0.459696 →0.360486 | [head_pressure_r3_b2_s1.json](head_pressure_r3_b2_s1.json); 0.24% difference is not meaningful evidence to replace block4 |

Ballast targets581696 resident bytes per L1 bank; it does not reproduce full40
fragmentation or replace loaded-model validation. The production head keeps the
legacy interleaved weight for many-row compatibility, adding about135 MB/rank.
The previous max-context free census is8.606 GB/rank, but final memory census
and the32K–64K smoke remain required. No capacity reduction is proposed.

## Full-path contracts and decisions

| Full-path boundary | Actual selected contract and evidence | Remaining stage-7 work |
|---|---|---|
| Embedding / first residual | BF16 ROW_MAJOR hidden-sharded `[248320,512]`/rank; one entry AG to selected replicated H2048. [Topology table](topology_audit.md), current `QwenModel.embed` | Final native width512 entry AG confirmed; cached height metadata caveat in final_profile_audit.md prevents payload inference. Replicated embedding remains untried, costs +762839040 bytes/rank and needs its own context census |
| Norms / residuals | Preserve replicated BF16 stream, selected L1 width shards at B1, compatible public row tuples at B>1; no inter-layer CCL. Terminal FP32 offset RMSNorm→BF16 unchanged | Native BF16 residual/L1 layout and FP32 terminal RMSNorm confirmed; final exact/all40 accuracy checks pass without changing compensated arithmetic or rejected residual families |
| LM head / logits | Vocab TP4, local N62080, DRAM-sharded reader3/block4 geometry above; BF16 logits copied into persistent sampler-ready `[1,1,32,62080]`/rank | Full40 top-k/B32/clean timing pass; native BFP8/HiFi2 FP32-destination reader3/block4 head282.987us and terminal norm22.689us at rank2 in reduced capture; context census passes; extra head storage disclosed in context_memory_audit.md |
| Greedy / sampled token | Existing common split sampler: physical top32 per shard; semantic greedy k1/p0/temp1; top-k/top-p remain supported; `tt_out_tok` writes persistent token input | Final sampler_acceptance.json and sampler_timing.json pass the greedy fixtures/strategies; real-terminal split0.487649ms, padded split0.521800ms, force-argmax2.669194ms. Native31-op split trace confirms local top32 and two candidate-only gathers; no generic TopK, force-argmax or full-vocab AG in selected decode |
| Token / position / RoPE | Persistent token, position, activity and RoPE inputs; device position/RoPE advance; nonblocking model then sampling replay | Final performance.json:127 queued model/sampling pairs, zero refresh/readback/recapture; observational delivery and actual teacher forcing reported separately |
| Cache / page tables | BF16 paged KV, page32, ownership `[0,0,1,1]`; FP32 recurrent and BF16 conv state. Changed-only `_host_pages` comparison before authoritative table update | All40 mixed replay and changed-only pages pass (30 changed tables, zero unchanged updates);32769/context65536 smoke passes (final/prefill_context32769.json); KV policy unchanged |
| Fixed slots / inactive rows | Explicit cache, positions, prompt lengths, batch/slots and active masks; request seeds/parameters/state and async output leases remain explicit | B32 passes at chip0 snapshot scope; all40 B3 mixed/inactive/sample-isolation replay and nine serving checkpoints pass; changed-kernel exact controls cover all four ranks |
| Prefill / host orchestration | Bounded executor retains arbitrary valid lengths and internal chunking; semantic config key deduplicates identical setup policy across layers | All40 replay covers resumed/multi-row prompts and stable program cache/traces; final TTFT boundaries pass64/255/256/257/1025/2049; prefill_performance_analysis.md supplies measured128-token and separate setup bucket tables with unresolved inline-role limits |
| Decoder persistent CCL | Shared serial `CollectiveWorkspace` direct-AR scratch, semaphores and EP outputs remain enabled. Inherited BF16+FP32 scratch1.5 MiB/rank; lifetime spans all traces | Final source/native audit confirms BF16 and FP32 direct-AR scratch inputs and serial model/sampler ordering; no claim of concurrent workspace ownership support |
| Sampler explicit persistent CCL output | `SamplerCCL.line_all_gather` accepts but does not bind `buffer_key`; captured addresses are stable, but explicit sampler AG output buffers are **not enabled**. A scoped ownership hook plus barrier-preserving broadcast AG was implemented experimentally and tested | **Measured rejected native partial-packet geometry.** [Watched raw gather](persistent_sampler_l1_watched_v3.diagnostics.json) has 3072/4096 mismatches on each rank: its own32 columns correct, remote96 unchanged. A 2048-byte tile cannot fill the native 4096-byte scatter packet; the writer has no tail flush. [Exact source/repro and unapplied repair](sampler_persistent_broadcast_rejection.md), [ownership/barrier audit](sampler_persistent_audit.md). No rejection based on timing alone; selected split sampling remains unchanged |

The explicit sampler-buffer limitation is distinct from the enabled decoder
scratch optimization. Neither captured-address stability nor a persistent token
output proves that all sampling CCL outputs use the explicit-buffer API.
The broadcast experiment is measured and rejected, rather than untried. Its
global 2048-byte fabric-payload workaround conflicts with the selected FP32
collective page size; there is no per-op packet-size API. Width64 padding at one
link is legal but doubles payload/output storage and requires removing four pad
chunks from each values/indices output (straightforward implementation: two
pads, eight slices, two concats). That workaround and the narrowly scoped native
one-tile packet specialization are **untried/unapplied**, with no claimed
speedup. See [anomaly resolution](anomaly_resolution.md) for retained failure
artifacts. Decoder persistent scratch remains enabled independently.

## Optimize checklist mapping

| Item / applicable advisory | Evidence or exact inherited decision scope | Status / action |
|---|---|---|
| Real shapes/weights/batch, traced warmed prefill+decode, context/nonaligned | Stage7 before and boundary JSON above; current head sweeps; inherited all40 contract | Final clean performance, boundary counterparts and reduced native profile pass; context smoke passes; B32 fixed-slot smoke passes |
| Topology before tuning; compare coherent collective families | Stage7 [topology_audit.md](topology_audit.md); [stage5 decisions](../optimized_multichip_decoder/optimization_decisions.md) include true sharded next-norm/residual consumers | Inherited family decisions retained; terminal/entry alternatives explicitly untried |
| Strongest correct baseline; reproduce final default | Same selected default under before harness; head controls use same real terminal hidden | Frozen final default21.416868ms matches candidate2 within0.04%; final correctness_acceptance.json passes |
| Shared qualitative suite, optimized-full-model onward | Prior six matched HF/TT prompts and AIME evidence in [validation plan](validation_plan.md) | Final bounded six-prompt and exact-ID AIME verdicts pass; see final evidence above |
| Reduced real-layer profile; watcher separate | Prior [profile_reduced_v4_drains](../full_model/performance/profile_reduced_v4_drains/) includes real0/3 terminal+sampler, CSV/provenance | Final reduced0/3 capture/report validates29440 executions,199 model+31 sampling ops/rank/replay,16 repeats/component; no all40 profiling; Watcher separate |
| tt-perf-report advice dispositions | Prior full-model v4 and stage5 B1/B32 reports | final_profile_audit.md dispositions cover all four advice strings: inherited sparse L1/subblock losses retain exact scope; selected fidelity preserved; isolated router L1/subblock controls now measured in router_advice_comparison.md; broader combinations remain untried |
| Roofline/device/end-to-end reconciliation; perf_summary.json | Useful-byte floor and historical stack estimate in topology audit; before clean wall metrics | Final perf_summary.json records separate regimes and queued layer estimate20.668310ms; native rank2 model/sampler work1.266784/0.436511ms is scoped to reduced capture; no additive claim for independent controls |
| Short/long/chunk-boundary TTFT, per-chunk kernel table | before_prefill lengths64/255/256/257/1025/2049 | After table complete; single samples range−0.030% to+3.428%, no causal/statistical claim. prefill_role_chunk_summary.csv covers measured128-token physical groups; setup buckets and unresolved roles separate. Long smoke is not repeated TTFT ranking |
| Dominant per-role configs, grids, blocks, subblocks (OPT-004) | Head table; inherited [local geometry audit](../multichip_decoder/local_geometry_audit.md), geometry/search.json and sparse_geometry/search.json | Current head explicit config verified in native metadata. Router input-L1/subblock controls pass exact tracked checks but give no win; broader full-config cross-products remain untried |
| DRAM-sharded decode, Blackhole readers1/2/3 (OPT-015) | Head sweep includes legal split one-reader repair and2/3 readers; inherited attention/shared reader screens | One-reader failure alone was not used to reject family. Per-role geometry remains explicit |
| Packed QKV vs split (OPT-001) | Stage5 `split_attn_l3.json`5.024359 ms loses to selected packed path; packed linear controls retained | Inherited decode role only; no new split default |
| SDPA/cache and reduced KV screen (OPT-002) | Selected paged SDPA/BF16; stage5 `kv_bfp8_l3.json` passed but0.23% decode gain with3.17% prefill regression | Preserve BF16; no new KV search or max-context claim |
| Residual layout through norm/add (OPT-003) | [residual_contract.md](../optimized_multichip_decoder/residual_contract.md), true sharded-family measurements | Preserve accepted replicated chain; no restored gather between layers |
| Logical batch vs tile padding (OPT-005) | Stage7 structural_components R1/3/31/32 BF16/FP32 with poisoned physical rows; existing fixed sampler32 rows | Component, public B32 chip0 inactive, and all40 mixed replay checks pass |
| Cumulative contract sign-off (OPT-006) | Policy table above, exact isolated integrated controls | Final cumulative correctness_acceptance.json and bounded native profile pass; context/review requirements remain separate |
| Output projection decompositions (OPT-008) | Stage5 direct AR, RS→AG, true hidden-sharded and repaired fused-output families, including next consumer; rejected slower or exact KV mismatch | Inherited decoder decisions only. LM-head row-parallel alternative untried, not declared impossible |
| Persistent decode CCL (OPT-009) | Decoder shared scratch enabled; candidate-local sampler ownership hook and broadcast init/final barrier protocol audited and hardware exercised | Sampler explicit-output broadcast **rejected by raw eager transfer gate** for native partial-packet data loss; exact repro retained. Explicit sampler buffers remain disabled. Padding/native repair alternatives are handoff items, not enabled optimizations |
| Packed/split MLP gate/up (OPT-010) | Indexed sparse packed GU and grouped fused SwiGLU selected; inherited shared packed GU/scalar | Further shared packed-vs-separate producer/consumer family remains inherited untried handoff; no fabricated stage7 comparison |
| Phase-specific activation shards (OPT-011) | Stage5 `batch_proj_decodeonly_l3.json` repairs prefill regression; head shard conversions counted inside component timing | Preserve selected phase policies; large-prefill2D geometry expansion untried |
| Precision×geometry (OPT-014) | Stage5 LoFi readers/blocks and HiFi2 singleton screens; current head sweep locks BFP8/HiFi2 | Different-fidelity screen does not reject untested selected-fidelity geometry; broad frontier explicitly out of scope |
| Tensor-group precision/fidelity, attention separate (OPT-007) | Stage5 natural-weight LoFi/HiFi2 comparisons; attention BFP4 failed real-output0.995 PCC; KV screen above; sensitive scan/recurrence retained | User requires preserving selection/rejection ledger. New expert/shared BFP4 frontier belongs to datatype-sweep, not this stage |
| Real activation precision evidence (OPT-012) | Natural B32 stage5 controls; reduced real AIME head hidden; saved real recurrent/scan fixtures | No synthetic-only veto or synthetic-only full-model acceptance |
| Dtype/fidelity reached actual ops (OPT-013) | Source/config and inherited runtime row evidence; head probe source hashes | Final native decode inventory verifies BFP8/HiFi2 head/full/shared, BFP8/LoFi active experts, BF16 KV and sensitive FP32/HiFi4 exceptions. Eager prefill inventory covers only668/12468 rows; no exhaustive inline precision claim |
| LM head/sampler/token feedback full decode gap | Head sweep, before counters, prior semantically greedy sampler comparison; no full-vocab AG default | Final correct sampler refresh passes; token-out is3.62% above queued isolated stack before terminal allowance; native selected split confirmed and clean sampler control is2.28% of all40 token-out |
| Active-expert MoE, exact nnz, GU/down tuning | E256/k8/I512; B1 indexed TP4 sparse, B>1 grouped EP4. Stage5 [MoE share](../optimized_multichip_decoder/moe_share.json), sparse geometry, natural B32 | Retain active selection and score-weighted combine; no dense all-expert substitution. B32 fixed slots and all40 mixed serving pass; dormant remap checks selected recurrent state on all four chips |
| Stress, fallback, Watcher, complete evidence audit | Structural tracked changing-input controls; direct recurrent Watcher component; inherited replay/fallback gates | Final all40 replay with Watcher10 passes17 cases/32 checkpoints; runtime guard/counters and reduced native audit pass; long-context passes; independent review clean-pass |
| T3K/Galaxy hazards and serving exceptions | Target is1x4 Blackhole; serial recovery evidence in work_log. No vLLM stage | T3K/Galaxy-specific and vLLM-only checklist rows not applicable; general device ownership/reset rules still apply |

Prefill DRAM interleaving, optimized SDPA, packed projection groups and explicit
important configs are retained contracts, not blanket claims that all kernels
have been exhaustively swept. The ordinary decode L1 sharding preference is
qualified by measured selected exceptions: sensitive ordered scan, sparse GU
DRAM input, and public singleton row interfaces.

## Multichip evidence mapping

| Required multichip evidence | Exact scope / remaining work |
|---|---|
| Baseline identity and strategy | Completed selected TP4/EP4 full-model at stage entry `ece17d22`; no new single-chip baseline reconstruction. [Stage4 mesh plan](../multichip_decoder/mesh_plan.md) and stage5 paired evidence document original single-chip comparison |
| Mesh, shapes, configs, sharding, padding |1x4 Blackhole; stage7 topology table extends decoder plan through embedding/head/logits/sampler. Logical vocab/sequence/batch remain separate from storage padding |
| MoE mapping/replication decision | E256,k8,I512; TP width128=4 tiles. B32 expected expert union63.7945%; retain grouped EP64 experts/rank for B>1 and indexed TP at B1. This is inherited measured hybrid, not a new replication proposal |
| Context byte arithmetic and largest capability | [context_contract.json](../context_contract.json) supports262144 aggregate tokens,32 slots, not32×262144. Prior all40 max-context/nonaligned acceptance retained; new head duplicate storage requires final census and32769 smoke |
| Nonaligned logical lengths | Stage7 before boundary rows plus inherited mixed/singleton/resume evidence; final optimized64/255/256/257/1025/2049 and resumed/mixed refresh passes;32769/context65536 smoke passes (final/prefill_context32769.json) |
| Norms/local heads/bias/CCL correctness | Retain replicated norm statistics, duplicated KV ownership and one-time score/scalar application from selected decoder; no new bias or reduction order |
| Coherent row-parallel families / reusable modules | Stage5 adapted distributed norm/fused AG-MM/MM-RS and true sharded streams have exact cache/performance decisions; stage7 head references legal DRAM-sharded reader family rather than reopening rejected arithmetic |
| Paired prefill/decode PCC and paged cache | Inherited paired single-chip/mesh gates; stage7 same-mesh exact component/integrated controls do not replace final HF full-model top-k refresh |
| Traced replay, determinism, state ownership | Existing split trace lifecycle retained; tracked structural and kernel controls are scoped below; final combined evolving all40 replay passes with cache9632 stable and captures2/releases2 |
| Warmed speedup/efficiency | Original mesh/single-chip efficiency remains stage4 evidence, not new full-model efficiency. Stage7 reports before/after same-mesh token-out, separate teacher metrics and useful-byte floor |
| Watcher and runtime fallback | Inherited reduced generator Watcher and clean guard; new direct-sum component Watcher. Final all40 Watcher replay covers fresh/resumed/mixed/inactive/sampled paths; exact changed kernels have separate all-rank Watcher controls |
| Profiling and CCL/semaphores | Final reduced report/native inventory complete; decoder scratch enabled, sampler native using_persistent_buffers=false confirmed; exact rejected broadcast-tail geometry retained; configure FABRIC_1D before mesh open |

## Completed repair evidence and final-source qualification boundary

[structural_components.json](structural_components.json) passes nine interfaces
with three changed inputs and four ranks, including nonzero physical padding.
`structural_permute.json`, `structural_padding.json`,
`structural_combined_l34.json`, and `structural_l3_b3.json` pass recorded exact
prefill/decode/state comparisons with tracking enabled. Their small timings are
diagnostics, not clean final ranking. The combined layer34 candidate is
0.731069→0.712077 ms in that tracked regime.

[recurrent_sum_acceptance.json](recurrent_sum_acceptance.json) is tracked plus
Watcher component evidence; [recurrent_sum_timing.json](recurrent_sum_timing.json)
is clean component timing: head-form0.030326→0.011170 ms, terminal-row-form
0.032134→0.009569 ms. Neither is a complete recurrence/layer/model claim.
`scan_{read32,read64,nop}_production_layout.json` are tracked exact current-oracle
controls including changed-input replay and required coverage. Their
`timing_eligible=false` measurements must not become headline speedups.
Earlier `scan_*_acceptance.json` controls have a layout-evidence limitation;
use the production-layout refresh for that contract. The precise custom-kernel
hypotheses and native source constraints are in [structural repair audit](structural_repair_audit.md),
[direct recurrence audit](direct_recurrent_sum_audit.md), and
[scan audit](scan_terminal_audit.md). These early controls were followed by the following concrete production selections:

| Change / exact evidence | Decision and scope |
|---|---|
| `scan_read32_timing.json`, current8×[32,128] output layout | Input111.770→99.893us, output66.846→53.680us, router35.892→35.094us. Narrow32 retained for router; no arithmetic/precision change |
| `scan_cache32_noclear_watched.json`, `scan_cache32_poison_watched.json`, `scan_cache32_noclear_timing.json` | Exact coefficient/product/K512partial/final and changed-input traces; poison proves unused coefficient padding irrelevant. Input93.846/output46.820us selected only for exact K512 input/output shapes. Cachedrouter37.480us and cachedK1024 103.044us lose to narrow32 and are rejected |
| `scan_cache32_watched.json`, `scan_cache32_timing.json`, elision controls | Full-cache zero fill costs more; reject. Terminal-store elision adds no material gain; not selected. Ordered additions/carry unchanged |
| `product_sum_dram_watched.json`, `product_sum_l1_watched.json`, corresponding clean timing JSON | Each528 exact checks; fused products plus ordered sum retain FP32/Q scaling. L1 plainK18.69→12.38us, DRAM scaledQ terminal18.79→13.63us. Integrated `production_cache_product_l0/l34.json` exact output/state/policy passes |
| `state_slab_r1_persistent_timing.json` | Complete selected recurrence66.064→35.971us; native BF16 consumption only, FP32 persistent state, exact16-then8 sum, bitwise scalar operand broadcast; no FMA/TF32 policy change |
| `state_slab_r{0,1}_{persistent,temporary}_128_watched.json` | All four rounded/unrounded×DRAM/L1 variants pass128-step exact state/output checks on4ranks. CPUexp captured gate values are shared-input component fixtures, not a production lookup replacement |
| `state_slab_integrated_l0_b1.json`, `state_slab_integrated_l34_b1.json` | Fresh tracked integrated exact output/state controls pass. `production_slab_l0_b3.json` also passes B3/length129/16decode exact controls; final all40 serving passes in final/serving_commands_all40_v3.json |

`state_slab_integrated_l0_b3.json` preserves an earlier baseline `position=None` harness failure; it is not a pass. The corrected production B3 artifact above is the passing control. `production_slab_functional_l0.json` and `production_slab_functional_l3.json` now also pass exact evolving clean layer controls; their FUNCTIONAL fields preserve the evolving layer controls. The current final queued engineering estimate instead uses final/queued_layer0.json and final/queued_layer3.json at0.586716/0.306684ms; do not substitute diagnostic12-sample medians or historical estimates.

See [state slab audit](state_slab_recurrence_audit.md) for corrected cross-producer
CB-counter and BF16-unpack-format bugs, single-producer CB26, and exact kernels.
Final production helpers preserve public shapes and unsupported-shape device
fallbacks. No Torch fallback enters the measured loop. [Runtime source review](runtime_contract_audit.md) and refreshed source hashes cover the selected slab path. Final clean counters and tracked all40 replay supply runtime evidence; final_profile_audit.md supplies separate reduced native dispatch/precision attribution.

## Binding stage5 policy and rejection ledger

The exact ledger remains [optimization_decisions.md](../optimized_multichip_decoder/optimization_decisions.md)
and [residual_contract.md](../optimized_multichip_decoder/residual_contract.md).
Selected replicated BF16 residuals, BF16 KV/CCL, BFP8/HiFi2 ordinary projections,
BFP8/LoFi active experts, FP32 sensitive recurrence/router and ordered scans
remain binding. A faster result does not authorize restoring a rejected policy.

- `batch_rs_ag_l3`, `batch_sharded_l{0,3}`, `sharded_bfp8_l0`,
  `autofix_fused_rs_l3`, `repaired_fused_bfp8_l3`, `ccl_bfp8_l3` reject measured
  alternate collective/residual families after adapting their next consumers.
- `attention_bfp4_l3` fails real-output0.995 PCC; `lofi_repeat_l3` gains only0.23%
  decode with3.27% prefill regression; `hifi2_repeat_l3` remains selected.
  `kv_bfp8_l3` similarly lacks material decode benefit and regresses prefill;
  BF16 KV remains unchanged. No maximum-context rerun is triggered by slab fusion.
- `batch_proj_l3` rejects decode geometry in prefill; repaired
  `batch_proj_decodeonly_l3` is selected. `split_attn_l3` and `shared_dram_l3`
  lose in their measured B32 roles. This is not a rejection of every unmeasured
  geometry at selected fidelity.
- The stage5 owner-carried linear speed limitation is historical context only;
  it does not waive the stage7 full-model floor. Stage6 HF-reproduced resume PCC
  limitation keeps its original scope and cannot excuse a new mismatch.

## Final serial queue and verified command surfaces

Use [final_gate_commands.md](final_gate_commands.md) and the executable
`tests/run_stage7_final_validation.py --output-dir FRESH_DIR` for the complete
serial correctness queue. Its dry-run is CPU-only. The runner stops on failure,
checks frozen source/idle device ownership, and includes exact-ID AIME, all40
replay/serving, B32 and an actually executed tracker positive control.
The all40 replay job uses Watcher10 and preserves its artifacts before the next
job. Performance, human output reviews, reduced native profiling and bounded prefill attribution have finished;32769 smoke passes. The abbreviated templates below are not a substitute for it.

These are **accepted CLI templates, not execution claims**. Run from the configured
checkout with the established lane environment, a fresh output location, one
mesh owner at a time, and an appropriate outer `timeout`. `D` below means
`models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model`; `M` means
`models.autoports.qwen_qwen3_6_35b_a3b.tests`. Correctness runs use
`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`;
clean timing uses tracking/watcher/profiler disabled. Do not mix regimes.

| Queue after candidate selection | Exact accepted command / constraint |
|---|---|
| All40 AIME24 chat-template prefill | `python -m "$M.run_full_model_readiness" --mode prefill --output "$D/after_prefill_accuracy.json"` |
| All40 teacher forcing | Same module with `--mode teacher --output "$D/after_teacher_accuracy.json"`; both top5≥98%, top100=100% |
| Six shared qualitative prompts | `python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/run_tt_controls.py --mode suite --output-dir "$D/qualitative_suite" --max-new-tokens 128`; current argparse explicitly accepts suite/autoregressive; suite default selects all six |
| Autoregressive evidence | Same script with `--mode autoregressive --output-dir "$D/qualitative_autoregressive" --max-new-tokens 128`; default selects q02; preserve rendered prompts and paired HF controls |
| B32 fixed slots/state | `python -m "$M.run_full_model_probe" --all-layers --generate --batch-size 32 --batch-probe --context 1024 --output "$D/after_batch32.json"`; runner chooses recommended trace reservation |
| Mixed/nonaligned/inactive/state replay | `python -m "$M.run_generator_replay_probe" --all-layers --context 2048 --section all --output "$D/after_replay.json"`; all40; separate same command with `TT_METAL_WATCHER=10`, profiler unset and distinct output |
| Tracker positive control | `python -m "$M.run_full_model_probe" --generate --positive-control --output "$D/after_positive.json"` |
|32769 nonaligned all40 context smoke | `python -m "$M.run_optimized_full_model_prefill" --lengths 32769 --context 65536 --output "$D/after_context32769.json"`; this driver **requires TRACKING=0** and no Watcher/profiler, prepares all physical variants, runs one long prefix then two resident traced steps. It checks tokens/positions/counters/cache-entry stability, not full last-page contents; add affected all-rank cache assertions if repairs change cache behavior |
| Final greedy sampler comparison | `python -m "$M.run_sampler_contract_probe" --mode acceptance --iterations 100 --output "$D/after_sampler.json"`; inspect same final head selection and greedy semantics |
| Clean all40 before/after counterpart | `python -m "$M.run_full_model_performance" --output "$D/after.json"`; diagnostics off; preserve separate teacher/model-only/sampled/readback metrics |
| Clean TTFT boundary counterpart | `python -m "$M.run_optimized_full_model_prefill" --output "$D/after_prefill.json"`; defaults are64,255,256,257,1025,2049/context4096; TRACKING=0 |
| Final reduced native profile | `bash models/autoports/qwen_qwen3_6_35b_a3b/tests/collect_full_model_components.sh "$D/profile_final"`; fresh directory only, reduced0/3, profiler-only; preserves commands/CSV/provenance and six report files; review advice, metadata completeness and per-rank spans |

`run_full_model_context_acceptance.py` accepts only `--output` and hardcodes
262144. It is not the32769 smoke command and should not be invoked for an
unnecessary maximum-context rerun. The final reduced Watcher battery must cover
all changed prepared paths, including multi-row/nonzero-start prefill; the
exact selected sections need checking against the final patch, not merely the
presence of a Watcher environment variable.

After these gates: reconcile stage7 `perf_summary.json`, final advice and runtime
fallback audit; refresh context provenance without lowering capability; put
before/after accuracy/performance and limitations into README/work_log; then
request the independent `$stage-review`. Review is not due while failed gates
or required evidence remain. Fix and rereview findings; only clean-pass or the
specified three-round pass-with-carried can complete review. Local stage-owned
commits and logged SHAs are complete, recorded in work_log.md; nothing was pushed.

## Stop Rule and untried items

Stop Rule3 applies after the user's explicit stage time/token-cap wrap-up.
The best validated version is retained; no further optimization experiments run.
Branches1/2 are not claimed. Untried performance items are handoff notes in the
work log. Correctness, required evidence, independent review and local commits
remain mandatory.

Untried or not yet finally resolved: native one-tile broadcast packet repair or
padding/compaction workaround for the measured-rejected explicit sampler AG
output candidate; wider selected-policy full-layer router core/subblock cross-products; broader sparse block64/grid/subblock cross-products;
replicated or batched embedding alternatives; head row-parallel/fused
norm-to-head topology; more head output-core/subblock/split combinations;
selected-fidelity B32 per-role cross-products; a public batched residual carried
through the whole stack; canonical-order fused AG-MM preserving paired KV;
shared packed-versus-split GU/scalar with gathered-input reuse; grouped expert
core/combine-buffer tuning; broader prefill2D/chunk geometry. Expected gains are
unknown until measured. Additional datatype frontier candidates belong to
`$datatype-sweep`; the selected decoder policy and rejection ledger remain
binding here.

## Final accounting and router-advice addendum

`native_accounting_derivation.md/.json` reconstructs layer-owned operation
ranges from source and ordered native rows, reconciles all16replays/rank, and
reports20.631607125ms same-rank all40 extrapolated operation work. The distinct
same-run reduced useful-byte ideal/native-work/fenced-wall triplet is
0.3500175625/1.7032948125/2.0154643753ms. These are not measured all40 device
critical-path time or host overhead. `perf_summary.json` records both scopes.

`router_advice_comparison.md/.csv` closes the exact L1/subblock advice with
three real-weight, synthetic-input isolated controls. Each has192 exact
all-rank checks in clean timing and separate tracking1/skip0/Watcher10 runs.
L1 and combined lose; DRAM/subblock2 has no demonstrated improvement. Initial
probe-only alias lifetime failure is fixed without acknowledgments/exemptions;
all six v2 artifacts supersede initial probe results. Production is unchanged.
