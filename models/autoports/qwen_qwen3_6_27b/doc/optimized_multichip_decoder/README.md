# Qwen3.6-27B optimized multichip decoder

Stage complete with independent round3 **pass-with-carried**. Final29 defaults, all28 timing jobs, all68 validation jobs and all six final profiles are complete. `final29_acceptance.json` records acceptance, remaining uncertainty, and local checkpoint SHAs.

Target: four Blackhole P300c devices,1×4 FABRIC_1D, tensor parallel on mesh axis1 with Linear collectives. The model is dense (48 GDN,16 full-attention layers); MoE does not apply. This stage contains decoder work only. Starting checkpoint is8502f42bca20dcad1507fbc8cebf7032220b7d8e; frozen stage4 decoder SHA082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44.

## Fresh default-path measurements

Milliseconds, serial full-mesh processes, same current harness/reference/native build for each before/after pair. Prefill is the arithmetic mean of three warmed complete calls. Decode is100 warmed nonblocking trace replays. Timing excludes watcher, profiler and allocation tracking. B1 rows below use the first prescribed alternating pair; the complete five-pair analysis is reported separately below. Final runs use no policy file, source override, single-device or replicated fallback.

| Kind | Batch | Logical prefix | Warm prefill before→after ms | Traced decode before→after ms | Final prefill/decode PCC |
|---|---:|---:|---:|---:|---:|
| GDN | 1 | 512 | 2.558 → 2.162 | 0.5739 → 0.5103 | 0.999976 / 0.999990 |
| Full attention | 1 | 512 | 1.735 → 1.628 | 0.4896 → 0.4112 | 0.999989 / 0.999978 |
| GDN | 32 | 33 | 11.530 → 5.880 | 0.7335 → 0.6190 | 0.999975 / 0.999980 |
| GDN | 32 | 513 | 59.082 → 54.428 | 0.7337 → 0.6191 | 0.999973 / 0.999989 |
| Full attention | 32 | 33 | 11.696 → 3.596 | 0.7004 → 0.4193 | 0.999892 / 0.999785 |
| Full attention | 32 | 513 | 44.742 → 35.450 | 0.7479 → 0.4462 | 0.999935 / 0.999937 |

Exact commands, all per-user/contribution/cache-state PCCs, source/native mappings and timing samples are the `final29_pair_*` and `final29_perf_*` JSON, `.command.json` and `.log` bundles, selected by `final29_plan.json`. Numerical threshold remains0.995. Their reference is the frozen single-chip **TTNN** optimized decoder, not HF. Independent HF output/contribution and resume/list checks are separate `final_validation29_contract_l{0,3}_tp4` gates.

The five independent B1 pairs have lower mean prefill for GDN2.395564→2.194060ms and full attention1.761020→1.701235ms. However, exact one-sided median confidence upper bounds are+0.037668ms and+0.057086ms. The predeclared95% zero-margin nonregression inference is **inconclusive**, not passed; all samples remain in `final29_b1_uncertainty.json`. No optional repeats or estimator changes were used. Round3 independently accepts this performance uncertainty as nonblocking carry under the owner's cap; it remains an inconclusive confidence result.

## Selected path and contracts

The selected path retains hidden-sharded BF16 residuals throughout the decoder stack: prefill[B,S,1280], compact decode[1,B,1280] per rank. The legacy decode[B,1,1280] interface remains supported. Norms and projections place their collectives inside a layer; the stack passes the existing residual tensor/address directly to the next layer. Public non-aligned lengths own internal padding, masking and slicing. Full details: [inter-layer contract](final29_interlayer_contract.md).

Kept changes include flattened multi-user prefill, persistent decode CCL buffers, packed GDN primary/correction and QKVG projections, repaired fused matmul/reduce-scatter placement, DRAM-sharded decode matmuls, local recurrence/affine work, and fused QK normalization/RoPE. BFP8 weights and KV remain default; BFP4 evidence is hand-off for the dedicated datatype stage. [Candidate decisions](final29_candidate_decisions.md), [final selection](final29_selection.md), and the initial [topology audit](topology_audit.md) identify comparisons and artifacts.

The final HF-contract gate exposed an inherited LoFi GDN attention failure: frozen stage4 reproduces the same S31 contribution row0.986014. GDN attention HiFi4 repairs it to0.999705 and passes the complete unchanged HF contract. Final29 uses that conservative precision correction for GDN; MLP and full-attention fidelity remain as selected. See [AutoFix](AUTOFIX_contract29.md) and [AutoDebug](debug_contract28/AUTODEBUG.md). The timing table above measures this repaired default, not an earlier faster LoFi candidate.

Native minimal fused normalization passed isolated/layer tests but failed64-owner L1 allocation. It is **off** in the final default. Row placement does not partition the lockstep allocator; [AutoDebug](debug_norm_capacity27/AUTODEBUG.md) reproduces the allocation ledger. DiT64-owner and ephemeral-scratch alternatives remain untried. No failed resource candidate is used to reduce public context.

## Validation, profiling and reproducibility

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-27b/lane-env.sh
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
python "$D/final29_evidence.py" run-perf
python -m models.autoports.qwen_qwen3_6_27b.tests.run_optimized_multichip_validation --prefix final_validation29 --run
python "$D/final29_evidence.py" run-profiles
```

These scripts refuse to overwrite existing evidence; use a fresh prefix for a deliberate rerun. The validation summary records all68 serial jobs passed: frozen TTNN references plus current TP4 correctness, actual-HF contract, five-layer stack, independent64-owner resources,32K context, B1/128-step and B32/512-step repeated stress, and non-aligned watcher cases. Minimum actual-HF token-contribution PCC across the checked length cases is0.997942 for GDN and0.997976 for full attention. Every device process is bounded. Watcher10 and strict allocation tracking1/skip-cache0 are separate from profiling. Resource tests share representative weights and do not claim full-resident64-distinct-layer device execution.

Six completed final-default profiles are in `perf/final29_profile_*`, including tt-perf-report prefill/decode tables and CSV, raw ops, metrics, dtype/geometry and roofline reconciliation. Advice is mapped to measured decisions or explicitly untried cap hand-offs in `final29_advice.json`. Final decode profile rows account for66–69µs of intra-layer CCL per step; stack boundary instrumentation reports no inter-layer CCL. The modeled decode traffic lower bound is43–45% of the instrumented host latency, not a70% claim or a replacement for uninstrumented timing above. Exact categories and limits are in `final29_profile_summary.json` and each `accounting.json`. The tool assumes8 workers for DRAM-sharded matmul despite the selected16-worker/two-reader configuration: its `Cores`, `FLOPs [%]` and compute-bound labels are invalid for those rows (including a175.65% annotation). Preserve raw reports; use the measured latencies and byte-based accounting. See [profiler limitation](final29_profiler_limitations.md). Native build/install receipt: `fused_rms_minimal_build27.json`; final native inventory: `final29_native_build.json`. The CPU/runtime audit scope is [documented](final29_runtime_scope.md), with48 negative controls and executed six-API sentinels for both kinds.

Public B1 context remains262144 with unchanged BF8 page32 KV layout. Stage5 checks32K context plus full-model capacity accounting; maximum-context proof is inherited from the accepted stage4 contract, not falsely rerun here. `final29_capacity.json` reconciles the passing final resource/context gates with the selected policy; the stage-owned context section is now validated.

## Stop decision and hand-off

The user explicitly ended optimization search at the stage time/token cap and instructed required gate completion. [Exact instruction](user_wrapup28.json) and [runner evidence](user_wrapup28_runner.txt) supersede the original optimization effort threshold. No70% roofline or8h device-time claim is made. Quality gates remain unchanged. [Untried hand-off](final29_handoff.md) records unfinished geometry, collective/norm, projection, recurrence and buffer experiments. Historical investigation text is retained in [README history](README_history_before_final29.md); the chronological [work log](work_log.md) preserves failures, commands and decisions.

The CPU artifact check uses `check_final29.py` to restore the planner’s package context without changing frozen sources/assertions. The original `final_gate.json` is preserved unchanged. `final29_postcommit_gate.json` passes26 groups including review/commit provenance and remains false solely for the B1 confidence inference. Independent round3 (`final29_review_round3.md`) accepts that uncertainty and the profiler annotation issue as nonblocking carry, with no Required Work. This is separately recorded in `final29_acceptance.json`; the failed statistical result is not relabelled. Local shared-code checkpoint is `3c8d8198482286af9c23688a1132d5caf4efa8f3`; model/evidence checkpoint is `421b68be97e410ac9d84619f2dae081d11447193`. No push occurred.
