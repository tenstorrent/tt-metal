# Qwen3.6-27B optimized multichip decoder

Stage 5 is in progress. No completion, final-default performance, independent
stage-review pass or checkpoint commit is claimed yet.

The starting implementation is the completed TP4 decoder at commit
8502f42bca20dcad1507fbc8cebf7032220b7d8e, file SHA256
082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44.
Target: four Blackhole P300c chips, 1x4 FABRIC_1D, Linear collectives on axis1,
two links. The model is dense. Full-model and vLLM work are excluded.

The unchanged acceptance bar is PCC >=0.995 for outputs, contributions,
per-user values and cache/recurrent state. Public context remains262144.
Non-aligned logical lengths must continue to pass. Initial topology audit is
in [topology_audit.md](topology_audit.md); commands, decisions and failures are
in [work_log.md](work_log.md).

## Untouched-path measurements

Serial, uninstrumented, 100 nonblocking warmed trace replays; prefill is the
mean of three warmed complete calls with synchronization at the boundaries.
Raw numerical and provenance records are before_l*_b*_s*.json and matching
.command.json files. All six rows pass the unchanged numerical gates.

| Kind | Batch | Logical prefix | Warm prefill ms | Traced decode ms |
|---|---:|---:|---:|---:|
| GDN |1|512|2.139|0.574|
| GDN |32|33|11.156|0.733|
| GDN |32|513|59.001|0.733|
| Full attention |1|512|1.618|0.490|
| Full attention |32|33|10.999|0.701|
| Full attention |32|513|44.630|0.749|

Candidate measurements are in family*_summary.csv and flat*_summary.csv.
They are provisional and must not be substituted for final-default results.
The latest source introduces an optional compact decode interface: per-rank
BF16 TILE [1,B,1280], preserved on output, so consecutive layers need no
batch compaction or expansion. The prior [B,1,1280] input remains accepted
and returns the same layout. Prefill retains [B,S,1280]. Setup-only
upload_decode_residual supplies compact inputs; no gather is added at layer
boundaries. Final contract and resource/context validation are pending.

## Investigation artifacts

- [Prefill movement diagnosis](debug_flat_prefill/AUTODEBUG.md) and
  [AutoFix record](AUTOFIX.md).
- [GDN recurrence diagnosis](debug_recurrent/AUTODEBUG.md), with static
  ownership/queue and traffic checks in that directory.
- family1.json, family2.json, flat1.json, family3.json: exact candidate matrices.
- candidate_commands.jsonl: serialized commands, UTC, source hash, elapsed
  time and exit code. Each run_job.sh job also records environment and HEAD.

The requested stop rule has not fired. Watcher, trace-allocation tracking,
stress, context/stack checks, fresh perf reports, independent stage review
and local checkpoint commits remain required before closure.

## Candidate evidence (not final defaults)

The current GDN candidate is `integrated8`; the full-attention candidate is `qknorm9`, which adds parallel compact Q/K normalization to `integrated8` with L1 MLP activation storage. These paths retain the hidden-sharded residual contract. Every row below passes the unchanged numerical and changed-input trace gates. These are candidate measurements, not final default-path numbers.

| Kind | Batch | Logical prefix | Warm prefill ms | Traced decode ms |
|---|---:|---:|---:|---:|
| GDN | 1 | 512 | 1.958 | 0.494 |
| GDN | 32 | 33 | 5.583 | 0.594 |
| GDN | 32 | 513 | 43.224 | 0.594 |
| Full attention | 1 | 512 | 1.699 | 0.443 |
| Full attention | 32 | 33 | 3.512 | 0.471 |
| Full attention | 32 | 513 | 35.339 | 0.522 |

The candidate uses owner-safe packed GDN primary/correction weights, local decode convolution, packed SwiGLU, local affine/gate fusion, persistent DRAM CCL buffers, larger prefill MMRS chunks, and batched full-attention RoPE. Tiled GDN prefill convolution retains packed inputs for B>1; B1 keeps its measured faster prior path. Certified batched cache fill owns a private page-table lease and falls back to ordered writes after exposure or during capture. B2 and B32 full-layer ownership pass all22 alias/remap/resume/trace checks under watcher and allocation tracking, alongside eight focused skipped-page cases. Final cumulative coverage remains pending.

`prefill_chunks7` completes28 jobs/376 projection configurations at64/128/256/1024/4096 rows. The inherited2D family wins the material large-row comparisons. `prefill_alignment8` adapts the temporal128 family with K128 short-capacity scheduling and bounded L1 blocks: all three capacities pass, removing the prior excessive single-token schedule and physical allocation failure. The adapted family remains slower than the current512 temporal chunk path. Full logs, CSV, source receipts and rejected configurations are retained.

Full-attention B1 prefill varies across repeated runs; ABBA cache-flag measurements show no systematic cache-flag penalty. Final matched default-path timing is still needed before declaring B1 nonregression.

Packed SwiGLU has64 exact watcher/trace-allocation cases over B1..32. Local affine/gate has28 exact watcher/trace cases across six batch sizes, zero/extreme controls and B3 one-core wraps. Focused probes do not replace final cumulative resource/context/stress checks.

`interim_profile4` holds complete TP4 CSV/advice tables for the pre-epilogue candidate. `advice5` tries all10 L1/minimal-matmul recommendations on the newer integrated family. Full-attention BF4 weights fail the unchanged KV-state PCC gate at both HiFi2 and HiFi4; MLP BF4 remains evidence for stage8 only. Native BF16 recurrence packing differs from explicit RNE; its later512-step actual-text stress passes (`nativepack512`), but final integrated selection and stress remain pending.

See `pending_requirement_matrix.md`, `candidate_search.csv`, exact manifests and `work_log.md` for work remaining. Defaults, final profiler/accounting, independent review and commits are pending; the original strict stop rule has not fired.

## Later measured candidates (resume18)

`epilogue16` keeps beta/decay fusion provisionally: warmed decode improves
0.494→0.484ms atB1 and0.594→0.579ms atB32, with all12PCC cases passing.
`fusions18_summary.csv` adds current short-capacity SDPA geometry and compares
complete decoder paths with/without the next fusions. All18cases pass.

| Full-attention workload | Control decode ms | Fused QK/RoPE decode ms | Gain |
|---|---:|---:|---:|
| B1/S512 |0.437504|0.411757|5.88%|
| B32/S33 |0.470333|0.419523|10.80%|
| B32/S513 |0.497227|0.446285|10.25%|
| B32/S33 real text |0.469897|0.419008|10.83%|

These are actualTP4 candidate measurements, not final-default acceptance.
QK/RoPE passes all32batches with actual NaN physical padding, changed inputs,
rebound addresses and strict trace allocation checks. Hidden-norm two-program
fusion passes96exact batch/value cases but has only~0.8%B1 whole-layer gain
and no usefulB32 gain; further synchronization adaptation is under test.

Native recurrent public-wave control passes18cases (B1/3/9/10/17/32 ×normal,
zeroQKV,zeroState), minimumPCC0.999836, with50queued changed updates and strict
trace allocation control. This does not reject or accept the compact prepared
native family; its coherent input/state/output adaptation is separate work.
Head-major recurrence→norm/gate preserves the same arithmetic and removes
strided readout consumption without restoring compact FP32 in between; broad
validation/timing are ongoing. Candidate BFP4 KV fails the first unchanged
cache-state gate (~0.985); real-text and packing-floor diagnosis are pending.
The default remainsBFP8 and publiccontext262144 remains unchanged.

`device_time_resume1908.json` records2.09776h conservative mesh occupancy,
counted once forTP4. Neither the original8h device-time alternative nor the
>=70%whole-layer roofline conjunction has been satisfied.

## Resume22 candidate and rejection evidence

The packed QKVG→Q/K norm+RoPE+V/gate family passes all32 batch sizes with watcher and trace allocation tracking. Uninstrumented complete-boundary median is28.024µs atB1 and36.425µs atB32, versus33.407/42.402µs for split inputs plus fused norm/RoPE. Whole-decoder integration is queued; these microbenchmarks are not final-default numbers.

The dispatch-admission hidden norm passes96 exact cases plus four delayed-receiver cases. Its standalone latency remains slower than the current native sequence (B1 33.310 versus23.495µs, B32 32.051 versus22.505µs). Whole-layer comparisons will determine the decision; no candidate is kept solely because it removes a launch.

Head-major recurrent readout is rejected on coherent-chain timing despite258 exact passing cases. BFP4 KV is rejected on all four unchanged whole-layer/cache gates and a packing-only diagnostic; BFP8 remains selected. The diagnostic source is the accepted frozen TTNN cache on recorded real-text activations, as clarified by `kv20_reference_provenance_correction.json`.

Paired `fusions20` TP4 profiler tables/advice/CSV show full-attention decode49→32ops and472.075→419.032µs kernel+gap. GDN hidden norm33→31ops yields only576.491→574.833µs. See `fusions20_profile_comparison.csv` and `perf/fusions20_*`.

The public native recurrent path fails a long repeated-input cross-persistence PCC check, and a zero-state per-head check reveals a distinct arithmetic cancellation case. These failures are retained, not accepted. Compact native BF16/FP32 families are under AutoDebug/AutoFix; neither has a production selection. `native_recurrent_precision_diagnosis.md` separates persistence from arithmetic evidence.

`runtime_audit_resume22.json` passes136 reachable functions with zero violations and negative controls. Final default runtime sentinel attachment remains pending. `device_time_resume1925.json` records2.347h conservative occupied time; the strict8h/roofline stop condition remains unmet. The current work matrix is `pending_requirement_matrix_resume1925.md`.

## Resume24 results

`gdn22` completes25/25 whole-layer passes; `attention22` completes12/12. CSVs `gdn22_timing24.csv` and `attention22_timing24.csv` record warmed prefill and traced decode. Column1 reduces GDN B1 decode0.484161→0.466628ms andB32/S330.578870→0.575519ms. Direct-packed QK/RoPE gives a small full-attentionB32 decode improvement but does not improveB1 decode over the split/fused control. No final-default claim follows.

Readout-only AutoFix (`native_readout_stress24`) passes eight zero-state/normal cases atB1/3/10/32, preserving original same-dtype native persistent-state bits and unchanged .995 per-head/user/state gates. First-step failing per-headPCC0.9887767 improves to0.9999997. Long timing/repeated-update evidence remains pending. Prediction-register reuse (`prediction_perf24`) is exact and changes isolated BF16-state recurrence52.717→51.401µs atB1 and131.811→126.912µs atB32; its combined column path still requires hardware measurement. Attention head-conversion/gating smoke passes exact native comparisons, poisoned padding and tracked changed/rebound/queued replay; wide coverage/timing remain active.

The fresh `device_time_resume2011.json` ledger records2.536226h conservative completed occupation. Running and timeout intervals with unknown endpoints are excluded. The original8h-or-roofline stop rule is still unmet. Independent final-validation harness review is adding provenance, receipt coverage and a paired original-TP4/default B1 prefill nonregression gate; this is preparation, not final acceptance.

### Resume25 status

`adapt24.sh` closed normally through all six traced-prefill workloads before its between-job STOP. The full-attention B1 quality probe initially failed a harness counter assertion: native single-user paged_fill_cache does not increment the batched ordered counter. The repaired probe instruments actual native fill calls during capture, still requires zero bulk calls, and requires the ordered counter for B>1. Fresh `traced_prefill23_l3_b1_s512_quality_v2` passes; no timed-forward algorithm or quality threshold changed. Eager/traced median prefill µs for GDN B1S512, B32S33, B32S513: 2305.475/1615.286, 5803.239/4526.670, 44179.652/43057.979. Full attention: 2014.700/1417.028, 3914.772/3303.584, 35419.489/35716.356. Thus traced prefill improves the shorter workloads, but full-attention B32S513 is slightly slower. These are separate traced-prefill experiments with reset/copy boundary, not replacements for final same-harness eager-prefill numbers.

`attention_gate_wide24` passes64 exact cases (all B1–32 × logical heads6/32, poisoned padding, changed/rebound inputs, queued replay). Native/direct canonical L1 gate medians µs B1/B3/B17/B32: 26.655/17.753, 26.781/18.135, 28.456/21.887, 20.147/26.759. A real row-band adaptation distributes each output tile across two disjoint half-tile writers. CPU ownership224 cases and watcher/tracker smokeB1/17/32 pass. `attention_gate_bands_perf24` gives native / one-band48 / two-band96 µs B1 26.652/17.759/24.202, B3 26.791/18.132/24.726, B17 28.459/21.889/28.712, B32 20.141/26.711/31.505. Reject the row-band adaptation on measured latency; do not claim B32 improvement. New off-default decoder hook preserves actual o_proj canonical input and excludes fused-MMRS restore for the direct-L1 family. Whole-layer comparison is queued in `attention_gate25.json`, both QK families and all four relevant regression/text workloads.

Joint output BF16 adaptation passes24 exact combine cases,14 actual-weight projection cases and2 actual B32S33 projection cases (`joint_output_*_v1`). The phase-specific full-layer matrix `joint_output23.json` remains queued. Prediction register retention passes isolated timing and changed-state tests; a new off-default hook composes its exact reader/compute transform with full column reader/compute/writer ownership. Combined hardware is queued, not claimed passed.

New coherent fused distributed norm→activation gather→actual projection-input/projection family passes first B1 TP4 watcher/tracker smoke `fused_dit_norm_b1_smoke25`. Broader B1/3/17/32, both kinds and both actual norm weights, separate quality/perf are queued in `norm_whole25.sh`. Inputs in these probes are labeled synthetic residuals with real model weights; no actual post-attention-residual claim. The separate native minimal fused RMSNorm family has a source-proven invalid stats iterator, mapping and phase-init bugs; installed AutoDebug report is `debug_fused_rms_minimal/AUTODEBUG.md`. AutoFix repair preparation is in progress without changing native source/binaries under active probes. Neither family is rejected on its first API failure.

`joint_conv_smoke25` failed the finite PCC input assertion after normal device closure. Investigation preserves poison/finite/exact gates and will record per-path diagnostics. No performance result or rejection follows from this first failure. Root owns the entire four-chip mesh and all jobs remain bounded and serial. Final defaults/gates/review/commit and the strict original stop condition are still outstanding.

### Resume25 — native AutoFix verified, convolution dtype repair

All mesh jobs closed before applying `debug_fused_rms_minimal/repair.patch`. Canonical build `cmake --build build_Release --target ttnn -j 12` and both runtime/pybind installation components completed; `fused_rms_minimal_build25.json` records all18 prior/new native sources and installed binaries, with complete build log. Old build3 binary hashes are now historical, not current final-build proof. The native repair replaces invalid iterator/core partitioning with bounded gathered-page mapping, admits one sender-local stats shard containing all rank tiles and one link, and replaces the invalid second kernel hardware startup with sender-local phase data-format reconfiguration. Independent AutoDebug/AutoFix and CPU mapping/admission evidence are under `debug_fused_rms_minimal/`.

`fused_minimal_norm_b1_smoke25` passes96 strict norm/projection comparisons, two changed banks, actual FP32 gamma rebinding, six trace allocation checks and effective positive control; mesh closed normally. First coherent norm/projection traced µs B1 input control50.828/146.846 versus minimal41.920/141.321; post control50.983/155.833 versus minimal42.166/150.733. These are within-layer boundary candidates, not whole-decoder or final-default results. Broader minimal and DiT cases continue serially with separate watcher quality and uninstrumented timing. No isolated native API error is used as a performance rejection.

Joint-convolution diagnostic v2 is explicitly INVALID as a quality gate: a diagnostic indentation defect omitted some per-user output comparisons, and an unequal-constant PCC diagnostic raised before complete capture. Preserve its failed artifacts; no pass claim. Corrected v3 records all12*B comparisons before failure and saves `joint_conv_smoke25_v3.first_failure.pt`. Native output and candidate corrected rest/history are exact, but candidate convolution output has1920/2560 mismatches and240 nonfinite values per rank. Source establishes that copy_init does not change unpacker dtype: FP32 joint-combine startup persisted while reading BF16 history/taps/token. Added explicit source-A format transitions before joint combine and before convolution. No arithmetic, rounding, state layout or quality gate changed. Fresh `joint_conv_smoke25_v4` passes bothB1/B3, changed/rebound inputs,50queued traces, watcher and allocation positive control. Wider timing/quality remains to run.

Independent hook audit finds selected attention gate admission/ownership and prediction-column composition consistent. Prediction transforms apply to every S1 call, including prefill/resume tails. Added constructor admission requiring packed batched convolution including B1 and cumulative-exp baseline for prediction, so an invalid policy fails before any convolution state mutation. Defaults still off. `remaining_bottleneck_audit25.md` identifies new traversal candidates and explicitly warns that existing S33 profiles and decode roofline bounds cannot explain S513 prefill latency. Fresh S513 profiles are queued before attributing any long-prefill saving.

### Resume26 — complete norm comparisons, remaining integration gates

All27 whole-decoder off/minimal/DiT comparisons pass unchanged output/contribution/per-user/state and changed-input traced replay gates (`norm_whole26_timing.csv`). GDN B1 regression traced decode is0.467117/0.450127/0.453307ms; actual-text is0.466958/0.450065/0.453140ms. GDN B32S33 is0.575952/0.565775/0.568183ms; S513 is0.575230/0.565905/0.568252ms. Full attention B1 is0.411646/0.403780/0.408923ms, B32S33 is0.419200/0.411387/0.412794ms, S513 is0.446352/0.439386/0.443497ms. Minimal is the leading norm candidate, not the selected final default. Its64-owner L1/resource feasibility, all-batch mixed-trace ownership, final context and final stress remain mandatory. Prefill sample variation is retained verbatim; this decode-only hook does not establish a prefill speedup.

Allfour integrated B1 norm stress50 watcher/tracker runs pass after generating explicitly matching frozen single-chip TTNN reference trajectories. These reference jobs are only oracles; each candidate executes the actualTP4 path. `prefill_qkv_smoke25` passes four exact B1/B3 × S33/S512 movement cases with changed inputs, queued replay and tracker positive control. Broad nonaligned geometry/perf and whole-prefill integration remain pending. The new root hook proposal is unapplied and separately CPU-reviewed; no default changed.

Long-prefill B32S513 profiles run serially with watcher unset. Native seed-only precision AutoFix is prepared in `native_seed_autofix25.md`: it mirrors the native decayed-state DST result into a direct-FP32 seed CB while retaining original FPU prediction/update math, and records all nine raw trajectory checkpoints before evaluating any failed gate. Hardware is pending.

Resume27: two-owner fused-norm probes pass both kinds/families; 64-owner proof remains pending. Native seed BF16 repairs the collected quality failure but is1.87x slower; FP32 still fails update quality and is1.89x slower (`native_seed_autofix26_result.md`). Off-default combine-output windows target measured706us/260MiB per-rank concat at B32S513; whole-layer trials pending. No final default or completion claim.

Resume27 admission/audit: native build27 installed; validation27_v4 passes21 native+1 separately classified framework rejection and8 healthy binding checks. All64 compact-prefill QKV cases pass watcher/tracker. Runtime static audit v2 passes154 functions/48 negative controls; final runtime sentinel remains pending. New prefill candidates are opt-in. Current continuation: `resume27_v4.sh`.
