# Optimized multichip decoder work log

Stage 5 started 2026-11-13 at 20:35 UTC from accepted multichip checkpoint
79582226 on main. Scope: four-chip decoder only. No full-model or serving work,
no push. Prior-stage untracked artifacts are preserved.

Installed optimize, multichip, tt-device-usage, stage-review and autofix contracts
apply. Both plugins are installed/enabled; environment.py exports validated.
Hardware admission: fuser reported no holders; bounded tt-smi listing found all
four Blackhole p300c devices (hardware_initial.log). Target logical mesh1x4,
p300_x2 descriptor and FABRIC_1D_RING. Device jobs remain serialized and bounded.

## Starting audit

The accepted source and advice-backed final layer0/1/3 profiler reports were
read before tuning. Residual is one complete BF16 2560-feature stream/rank;
no inter-layer collective. Meaningful kinds: GDN0, PLE/GDN1, QSA3. EP4 active
experts are retained. Existing BFP4 expert storage is the accepted physical-fit
contract (BFP8 all-layer weights do not fit); this stage does not newly select
BFP4. Exact sensitive projections retain their numerical contract.

Independent CPU audit is writing operation_topology_audit.md. Highest-cost
candidate: parallelize QSA ordered indexer K partials while preserving pair and
partial accumulation order. Column partition alone would reduce core count
without reducing serial work, so the candidate distributes independent K blocks.
Other families: async/persistent ordered gather, packed shared gate/up,
GR injection geometry, lower-movement residual next-consumer path, DRAM shards
and precision. All kept changes require full target-mesh evidence.

## Status

Selection frozen at21:38 UTC under Stop Rule2. Stage complete: final default gates, profiler reports and independent stage review pass; local checkpoint recorded below. Historical progress entries below describe their time of writing.

## Initial serial measurements

Fresh pre-pass timing: GDN0 prefill33/B1/B32 =21.4189/2.26639/57.8812ms;
PLE/GDN1=26.7083/3.43220/95.1313ms; QSA3=80.1750/4.75097/131.1169ms.
`before_l{0,1,3}` JSON, logs and provenance use the accepted clean harness,
32 trace replays, state restoration outside timing and no diagnostic tracking.
All pass. Same regime as stage4 final numbers; full acceptance remains separate.

`async_l0` keeps ordered BF16 transport/FP32 sum and unchanged residual contract.
PCC unchanged, but B1/B32=2.30024/58.7469ms versus2.26639/57.8812ms.
Persistent-output variant remains to be tried; this does not reject that family.
Shared packed gate/up at HiFi4/HiFi2/LoFi passes whole-layer PCC, gives roughly
0.8% B1 improvement but approximately0.2% B32 regression; separate LoFi has no
material benefit. No selection yet. Exact candidates/provenance use the named
`shared-*` artifacts.
`inject-10_l0` partitions10240K into10 independent1024-wide FP32 partials,
then sums and casts BF16. It passes identical short PCC, B1=2.17083ms,
B32=54.8231ms (4.22%/5.28% faster). Further geometry and full contracts pending.
Parallel ordered-indexer component candidate is running against actual saved
HF activations and exact original BF16/FP32 controls. This component probe uses
one chip; the candidate must subsequently pass and be measured inside the
four-chip decoder before any stage claim.

The first `parallel-indexer_l3` experiment did NOT reach the candidate:
Multichip uses FusedOrderedIndexerProjection, which replaces the original
constructor. Its unchanged timings triggered a source audit. This run is
excluded as optimization evidence. The exact component test remains valid,
but the parallel kernel must incorporate the existing fused valid-row reader
and compute optimization before binding the actual multichip path. In particular
Q1 original full-tile arithmetic is much slower than the fused active-row path.

Persistent async gather (same ordered sum) passes but remains slower:
B1/B32=2.27989/58.0905ms. BFP8 async payloads cost2.33470/59.6262ms;
whole B32 PCC0.997227, diagnostic rows5/14/30 below0.99. No catastrophic
collapse; real-weight full decoder is slower and less accurate, so global
payload conversion is rejected. Separate subgroup candidates are still open.
Reset1 followed the failed multichip run: fuser empty, bounded reset/list exit0,
all four devices returned, actual collective smoke passed. The first smoke
command had a host-only indentation typo before importing TTNN; its log is
preserved alongside the successful retry. No process was killed.

## Topology and routing follow-up

`parallel_indexer_v2_l3` reaches the multichip QSA binding and passes unchanged
short PCC: prefill33/B1/B32=42.36697/3.09050/78.06350ms. Combined with inject10,
`combined_control_l3`=42.30587/2.99593/75.12868ms. The later
`parallel-indexer-rows` binding explicitly enables valid-row compute/copy-once;
`parallel_rows_l3`=42.28896/2.99712/75.16408ms, identical PCC. This second binding
fixes an experimental flag ambiguity, not an additional measured gain.

Pipelined PLE reads, composed with inject10: batches16/32/64 give B1
3.25050/3.24242/3.24216ms and B32 89.03596/88.86462/88.75695ms. All pass;
complete awkward-shape, watcher and tracked integration checks remain pending.

Real prompt fixture uses32 distinct checkpoint embedding rows and the exact HF
first-layer four-stream expansion. Layer0 is actual first-layer input; later
layer use is explicitly shape-faithful routing stress, not claimed full-stack
activations. Fixture/reference tensors stay outside git. Layer0 B32 has32
unique routing rows and103 selected experts; active execution is retained.
Isolated actual-intermediate MoE trace is2.21568ms (3.8635% of57.34894ms step).
B1 MoE=.58798ms (25.9335%). `real_mesh_before_l{0,1,3}` records all kinds.
First layer1 reference attempt hit host validation because new fixture token
words were rank1; corrected to [1,1,16], rerun `real_baseline_before_l1_v2`
passed. This was a harness defect, not a numerical rejection or device hang.

`fused_family_{HiFi4,LoFi}` passes Q1,2,31,32 and two changed inputs. Scope is WO
producer plus residual write plus entire next GR, without restoring the old
stream contract. Exact per-family traced and eager tables are in JSON. At Q1,
HiFi4 ordered BF16=.7177ms, RS=.5836ms, fusedRS=.6016ms, AG-MM=.5948ms,
fusedAG-MM=.5835ms. Eager timing reverses the ranking. Whole-layer adaptation
and comparison with the improved injection control are necessary before a
selection; the lower-movement family is not rejected on eager timing.

DRAM seed probes pass all five real local-weight roles. They include activation
reshard, output restore and packed split costs; inputs are seeded stress, so
precision decisions require real-prompt decoder follow-up. Expanded geometry
attempt `dram_geometry_qsa_v2` reaches physical L1 limit at split QSA block40,
8 storage shards, one reader:1998784B requested >1572864B. Prior blocks pass.
This is not a family rejection. Retry bounds blocks to1,2,5,10,20 and sweeps
8/16 storage shards and1/2/3 readers. The first sweep launch was host argparse
error (list syntax), retained separately. Reset3 follows the device API failure;
reset/list and actual four-rank collective smoke are recorded separately.

`pipelined_lookup_component` passes UINT32/BF16/FP32, logical16x20,17x33,33x65,512x20, batches16/32/64, exact changed-input traces with program-cache tracking. `bfp8_kv_component` passes the selected-row BFP8 unpacker on all4ranks with page permutations and changed IDs. `parallel_indexer_all_counts_v3` passes all Q1..32 across six actual HF windows, exact BF16 and pre-cast FP32, cross-shape/replay tracking. These remain component evidence, not meshdecoder performance.

Integrated initial defaults now select injection10, parallelordered indexer, and PLE readbatch32. No KV or residualownership change. DRAMgeometryv3 completes all five roles with1/2/3 readers and8/16 storage shards; expanded tables are JSON/CSV. HiddenQSA first attempt failed host Shape slicing before outputcomparison; tuple conversion fixes adapter. Reset4/list/smoke follows per multichip skill. No hang or kernel failure claimed.

### 21:28 UTC: real projection and coherent-layout selection

The current default stream-per-chip residual is retained provisionally: complete real-input QSA hidden-layout RS and AGMM candidates passed but B32 measured 76.5484/76.4509 ms versus current stream control 75.1224 ms. Both carried the candidate residual through GR and MoE without immediate restoration. Fresh fused-family component runs use the best injection10 control and verify trace against eager, in `fused_family_bestcontrol_{HiFi4,LoFi}.json`.

Expanded GR full-shard block40 sweep completed. All ten precision/activation component sweeps completed; these use real weights and seeded normalized inputs and do not decide model precision by themselves. Real decoder shared HiFi2/LoFi passed, but neither improved B32 over the HiFi4 packed-rows candidate. Real GDN HiFi2 passed. GDN LoFi output whole-PCC passed (short minimum .99932766, B32 .99907094); the inherited probe returned failure because one diagnostic row was .99447459, below its legacy per-row .995 check. State parity passed. This is not a whole-output accuracy rejection; the user specifies whole-output .995 with per-row diagnostics. The probe acceptance expression is being corrected to that stated contract, with all diagnostic rows retained. Its B1 2.16331 ms is not a material gain over compact HiFi4 2.16968 ms. Reset5 follows the failed multichip run; no process was killed. Exact commands and list/reset/smoke results are in reset5*.json/log and mesh_after_reset5*.

### 21:34 UTC: real dtype/CCL decisions and final selection setup

BFP8 KV real QSA passed whole-output/state checks but B1 latency was3.36262ms versus~3.000ms current control; B32 was75.4865ms versus75.1224ms. BF16 KV dtype/layout stays selected, so no new maximum-context run is required. Targeted MoE/QSA BFP8 CCL whole-layer runs passed but provided only noise-scale gains (54.2694 vs54.2817ms,74.9954 vs75.1224ms) and additional quantization. BF16 CCL transport retained. Real BFP8 activation trials passed but did not beat BF16 whole-layer controls materially.

Real BFP4 attention failed whole-output PCC: short minimum .96450335, B32 .98230147. This is an actual real-weight/input failure, distinct from the earlier synthetic probe. Reset6 and successful collective smoke are recorded. The compact QSA projection then passed, improving B32 from74.707ms (sliced/concatenated DRAM candidate) to74.2685ms. Shared packed rerun failed at Python import due to concurrent CPU harness editing, before device open; its source error is preserved in the first log and a separately named v2 retry is prepared after compilation. No device work was dispatched in that host-only failure.

## Final selected policy and freeze

Source formatted successfully (`precommit_final_source_v2.log`) before final measurements. CPU admission checks pass6 tests (`final_cpu_admission.log`). Selected DRAM roles: GR BF16/HiFi4 cores8/readers3/block5; compactGDN ZAB BF16/HiFi4 cores8/readers2/block2; shared packed BF8/HiFi4 cores8/readers3/block5; compactQSA QGate/K/V BF8/HiFi4 cores8/readers3/block2; WO BF8/HiFi4 cores8/readers3/block2. Remaining QSA attention uses HiFi2, independently from those HiFi4 projections. One shared weight representation serves both phases. Construction directly prepares selected weights and retains no native duplicates. BF16 activation and CCL transport, BF16 paged KV and stream-per-rank layer boundary remain unchanged.

The final combined candidates pass all representative kinds. Final default is measured separately with `plans/final_all.json`, serially: real fixture performance/MoE, tracked changing-input contracts,32/1024 matched timings, watcher10 alltails/stack,32769 full-stack reservation smoke, separate profiler captures and allocation tracker positive control. B32 guard now enforces the user-specified whole-output.995 bar; row diagnostics remain visible. Existing finite/state/exact-replay/tail/stack gates are not relaxed. Changed-batch validation additionally alternates hidden input and PLE token payloads in fixed buffers, with eager output/state references prepared before capture.

Selected all-layer persistent-weight increase is506757120B/rank; final long-smoke reservation is31710200640B/rank. Runtime planning envelopes remain1GiB(default)/2GiB(stress). Bounded stack IO remains required; the long-smoke allocator reserves resident state/weights before IO, not a separate2GiB runtime allocation. No reduced context or full-model capacity claim is introduced.

# Not tried (stop rule)

The owner records optimize Stop Rule2: the last two retained QSA refinements
each improved the target B32 workload by less than3%.

| Retained refinement | Before B32 ms | After B32 ms | Gain | Evidence |
|---|---:|---:|---:|---|
| Compact QSA projection output | 74.706688 | 74.268517 | 0.5865% | `dram_ablation_qsa_l3.json` → `dram_compact_qsa_l3.json` |
| Remaining QSA attention HiFi2 after role composition | 72.503128 | 71.619102 | 1.2193% | `dram_final_candidate_l3.json` → `dram_final_candidate_hifi2_l3.json` |

The selected roles are now integrated into the default. Other role integration
composes already measured choices; it does not introduce an unmeasured new search.
Both accepted optimizations preserve the gate contract. Candidate clean timing
is not final-default reproduction or final stage acceptance.

The following are performance handoff notes bounded by that rule, not blockers
or claims of measured rejection. Expected gains are unknown unless specified.
Do not repeat completed async/persistent/global and subgroup CCL, DRAM reader/
geometry/packing/fidelity, activation8, real KV8, or coherent QSA hidden-family
experiments merely because their alternatives were not selected.

| Untried extension | Existing evidence and next useful experiment | Expected gain |
|---|---|---|
| Hidden residual for PLE/GDN with sensitive exact WO | Whole-QSA native RS/fused AG-MM loses B32 despite component gains. A future extension must retain hidden slices through exact GR, norm, residual and the next decoder; no immediate restore to the old stream contract. | Unknown; QSA evidence is negative for the tested whole paths |
| Other whole-layer fused MM-RS combinations | Native/fused producer families pass through next GR in the component; whole-QSA native RS and fused AG-MM were integrated. Fused MM-RS with a different full-layer policy remains a distinct extension, requiring best-control B1/B32 and state gates. | Unknown |
| Packed-only active-expert execution | Current gate-selected execution remains correct. Dual expert formats are physically excluded by the prior per-device ledger; a packed-only implementation may change dispatch/layout cost without storing both formats. Keep selected experts and measure diverse B32; dense all-expert work requires evidence. | Unknown; use final MoE share to bound benefit |
| Additional phase-specific weight/layout or pipeline schedules | Packed shared now uses one immutable policy in both phases after measuring separate/phase-specific candidates. Further schedules need complete prefill/decode and persistent-memory accounting, not isolated MM speedups. | Unknown; current role gains diminish |
| Additional subgroup CCL dtype combinations or worker/chunk/link tuning | Global BFP8 regresses; real MoE/QSA subgroup gains are below0.2%; persistent ordered async regresses. New combinations must include casts, buffers and actual consumers. | Unknown; tested subgroup gains below0.2% |
| BFP8 KV at a long-context bandwidth break-even | Exact selected-row unpacking and short whole decoder pass, but B1 is12.09% slower. A future long-context branch must prove benefit, cache semantics and changed-format maximum context before selection. | Unknown; short-workload result is negative |
| New graph fusion/layout opportunities from the final selected-default profile | Starting material topology families were audited and tried. Preserve final advice output and map each new recommendation to existing experiments or a precise future experiment; do not invent a rejection from source/API availability. | See final profiler findings and measured policy readback |

Additional final-profile advice under **Not tried (stop rule)**:

- GR-only HiFi2 in a real whole-decoder comparison; geometry/fidelity component sweeps exist, but this dedicated integrated trial does not. Expected gain unknown.
- Shared-down and shared-route: L1 input, legal output subblocks and independent fidelity through their consumers. Their combined decode kernel cost is about 29 us per layer before conversion costs.
- Active-expert weighted sum: internal K padding for block2, L1 input, output subblocks and fidelity while retaining selected-order reduction. The current decode kernel is about 3.5 us.
- PLE key/value: pack the two same-embedding projections and compare L1 input, subblocks and fidelity through both consumers. Combined decode kernel time is about 101 us before added splitting/layout costs.
- Further prefill scheduling/fusion beyond current eager segmentation. Expected gain unknown; profiler drain gaps are not a measured trace bottleneck.

The following are **mandatory completion work, not deferred optimization ideas**:
final-default PCC and warmed prefill/traced B1/B32, actual dtype/fidelity and
advice-enabled profiler tables/CSV/provenance, nonaligned tails and32K–64K context
smoke with unchanged BF16 KV, selected-memory/context ledger, changed-input and
mutable-state tracked replay, runtime fallback audit, watcher-clean stress,
independent stage-review with fixes/rereview, and local stage-owned commit SHAs.
Any failed gate still requires repair. This document neither marks the stage
complete nor overrides the user's requirement to persist through failed gates.

## Final gates: correctness and context

All final real-fixture timings pass. Default B32 is52.452465/87.054184/71.598373ms for L0/L1/L3. All three tracked contracts pass with16 changed-input B1 and16 alternating-input B32 replays, exact own-eager output/state equality, all32 logical tails, runtime fallback guards, and two-layer boundaries without conversion. All three watcher10 batteries pass. Custom indexer/pipelined lookup sources exactly match their earlier all-shape changed-input component checks (`component_final_source_consistency.json`).

Final32769-token public prefill+three continuation steps pass at updated31,710,200,640B/rank planned reservation. Actual banked reservation is31,710,208,512B for PLE/GDN and31,710,209,024B for QSA. Whole-output minima are.9998434904 and.9991781194 respectively. Context262144 and unchanged BF16 page32 KV are preserved; this is a decoder smoke with all-layer reservation filler, not a full-model instantiation or measured full-stack runtime reserve.

## Final profiling and evidence assembly

All21 final plan jobs finish exit0, including all three profiler captures and the allocation-tracker positive control. Advice-enabled table/CSV generation exits0 for prefill and decode of each layer kind. The offline evidence collector returns `evidence_ready_for_independent_review`, issues0, with identical final runtime/probe source digests across every gate. Final fuser reports no device holders and bounded tt-smi listing returns all four p300c devices (`final_hardware_health.json`). No final reset or recovery was needed.

Complete raw captures remain on persistent storage under `/home/ttuser/qwen38-flash/optimized_multichip_profiles`; local logs/CSV copies remain intact. `artifact_storage.json` indexes large files. The large QSA prefill text table is also preserved as ordered portable parts to satisfy the repository500KiB file limit. No raw evidence is compressed, deleted or newly hashed. Source and command provenance from the established runner is retained.

## Final advice review handoff

Final raw rows confirm the selected DRAM dtype/fidelity/layout/reader/block policies and QSA SDPA HiFi2. See `profiler_findings.md` and `measured_policy_rows.json` for exact rows and the advice-to-experiment map. Additional PLE key/value packing uses the same embedding input and remains untried under Stop Rule2 (about101us summed decode kernel time before split/layout costs). Remaining small shared-down/router, expert-reduction and PLE layout/subblock/fidelity suggestions are precise handoffs, not measured rejections. Precommit normalized whitespace in derived text tables only; measured runtime sources did not change.

## Independent stage review

Fresh xhigh round 1 returned **clean-pass**, no Required Work, in
`stage_review_round1.md`. The reviewer independently checked before/final source
comparability, final timing samples, actual four-rank profiler policies, trace
lifetimes, fallback/watcher gates, active routing, and context arithmetic.
`before_source_delta.patch` and `before_source_explanation.md` explain the exact
shared-expert helper extraction used by real-before runs; the reconstructed
source exactly matches their recorded digest. No runtime repair or hardware
rerun was required by review. Final precommit and collector checks pass.

The review's future full-stack memory/trace-lifetime risks and performance
handoffs remain documented; they are not decoder acceptance failures. Only the
stage-owned files in this repository are included in the local checkpoint.
Prior-stage untracked fused-decoder evidence remains untouched. No push.

Commit-hook note: after staging new files, clang-format normalized the rejected
BFP8-KV candidate kernel `tt/kernels/local_selected_paged_kv_bfp8.cpp` (spacing,
line wrapping and braces around existing single statements). This file is not
reachable from the selected BF16-KV default. The measured default and all its
kernel sources remain byte-identical; candidate measurement provenance retains
the original source digest. The formatter-only change needs no hardware rerun
or new independent review round. The first commit attempt stopped at this hook;
the formatted file is included in the retry.

## Local checkpoint and stage completion

Repository: `/home/ttuser/qwen38-flash/tt-metal`; branch: `main`.
Implementation and evidence commit: `d103c86fc35f59f9d10f1e73daef969f92ffe258`
(`Optimize Qwen3.8 Flash Next four-chip decoder`). Its full commit hook set
passes. A following documentation-only commit records this SHA and acceptance
status. No push was performed. No full-model or vLLM work was started.

Stage 5 is complete with independent round-1 `clean-pass`, unchanged accepted
whole-output gates, final-default performance reproduction, complete profiler
and watcher evidence, preserved logical/context contracts, and Stop Rule2
handoffs. All stage-owned runtime and evidence changes are committed locally;
large raw captures/logs and reference tensors remain at their recorded local
persistent paths. Unrelated pre-existing fused-decoder artifacts are preserved.
