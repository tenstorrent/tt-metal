# Recent graph-fusing evidence reconciliation

This update reconciles raw artifacts; it does not complete the fused-decoder stage, establish the optimize Stop Rule, or waive any final gate. All timings below are layer 0, length 128, context 512, with 10 warmed prefill samples and 60 traced warmed decode samples. Artifact provenance records each source hash. These are sequential candidate measurements, not a randomized paired experiment.

| Candidate artifact | Before decode (ms) | After decode (ms) | After prefill (ms) | Decode reduction |
|---|---:|---:|---:|---:|
| [recurrent_boundary_fused_0.json](../recurrent_boundary_fused_0.json) | 1.635427 | 1.617843 | 83.760581 | 1.075% |
| [recurrent_l1_0.json](../recurrent_l1_0.json) | 1.617843 | 1.560814 | 83.973084 | 3.525% |
| [recurrent_l1_prefill_0.json](../recurrent_l1_prefill_0.json) | 1.560814 | 1.560764 | 78.832644 | 0.003% |
| [norm_residual_fused_0.json](../norm_residual_fused_0.json) | 1.560764 | 1.552968 | 78.887222 | 0.499% |
| [norm_scalar_fused_0.json](../norm_scalar_fused_0.json) | 1.552968 | 1.544202 | 79.140272 | 0.564% |
| [compact_rm_gather_0.json](../compact_rm_gather_0.json) | 1.544202 | 1.506119 | 78.959430 | 2.466% |
| [compact_decay_0.json](../compact_decay_0.json) | 1.506119 | 1.482640 | 79.128677 | 1.559% |
| [direct_projection_slices_0.json](../direct_projection_slices_0.json) | 1.482640 | 1.472941 | 79.187016 | 0.654% |

All eight short measurements report identical prefill PCC 0.9999717836843032 and traced decode PCC 0.9999857526339517, deterministic output, and a passing runtime-fallback audit. The prefill-only L1 change is kept for its 83.973084 to 78.832644 ms prefill improvement; its roughly 0.003% decode change is not a meaningful decode gain.

The recurrent-boundary microprobe reports all six complete multiply comparisons bitwise equal. Its initial failed diagnostic is preserved: an inserted negative zero differed in the standalone round intermediate, while the complete old and fused multiplies matched. The revised probe reports this distinction and CPU flush-to-zero behavior explicitly. This is an expression-equivalence clarification, not a decoder PCC relaxation.

RSUB and scalar-chain actual-method norm probes each check 17 sensitive rows from three seeds, with zero BF16 norm-code and FP32 mean-code differences. Compact coefficient-gather diagnostics compare the entire adapter including conversions: compact TILE was rejected by integrated timing, while compact ROW_MAJOR was retained. The decay-table proof checks all 32 × 65536 FP32 output codes; the separate device index probe checks all 65536 index codes, and integrated stream tests cover the assembled decoder.

Latest direct-slice qualification artifacts are now complete for all three seeds. Each contains one 8192-token decode run and a 33-token continuation after a 128-token prefill. These are single-repeat, single-capture qualification results; `repeated_bit_equal=true` in a one-repeat report does not establish a multi-repeat acceptance test.

| Seed | Minimum output PCC | Minimum contribution PCC | Gate |
|---:|---:|---:|---|
| 913 | 0.996662946692 | 0.996216457546 | pass |
| 42 | 0.996846472554 | 0.995666139992 | pass |
| 154 | 0.996650133902 | 0.996095021285 | pass |

Exact stress artifacts: `direct_projection_slices_seed913_results.json`, `direct_projection_slices_seed42_results.json`, and `direct_projection_slices_seed154_results.json`, all in this directory. Earlier retained rows in `pattern_assessment.json` name their own stress evidence; prefill-L1 qualification is carried by the later RSUB source, and scalar-chain qualification by the later compact-RM source. These combinations are stated rather than claiming missing standalone runs.

## Pattern coverage and hand-off

Every named pattern family in the installed graph-fusing skill has an assessment entry. Applicable dedicated operations have historical measurements, current measurements, or an explicit contract blocker. Distributed RMSNorm, RepVGG branches, spatial reductions, convolution bias/adjacent scale, and BatchNorm folding are structurally absent. Transpose-plus-matmul has no remaining runtime RHS transpose to fold because weights are prepared in setup. Slice-after-matmul has no discarded useful projection channels; direct consumer slicing removes reconstruction rather than changing model dimensions.

The historical native KDA adapter gap is now closed by measured current-source output-only controls, described in [the adapter memo](AUTOFIX_native_output_only_candidates.md). Retaining exact state updates, corrected UINT32 KDA and separate chunk-GDN adapters pass short PCC/determinism checks but take 1.584314 and 1.642141 ms decode, respectively, versus frozen 1.455763 ms. Both are rejected on complete latency; no long-stream eligibility is claimed. The original KDA INT32 setup validation failure is preserved separately.

Other exact blockers are already recorded: native recurrent GDN and recurrent matmul change state precision/order; native fused MoE hardcodes BFP4 weights; DRISC prefetch support is unavailable on the recorded hardware/firmware; newer fast-reduce output-dtype keywords are absent from the installed binding. Native output-gated norm decode adapters and compact TILE coefficient gather have measured losing controls. No new unmeasured scalar or direct-slice candidate is being mislabeled as a completed final gate.

Final source-matched repeated stress, watcher-clean correctness, representative full-attention coverage, advertised context/tail checks, refreshed profiler tables/CSV, final performance ranking, stage review and local commit records remain the root-owned completion work. The latest two decode improvements are below 3%, but that alone does not satisfy the Stop Rule: neither 70% of an honestly defined useful-work roofline nor eight cumulative device hours is established by this reconciliation. Do not substitute elapsed wall time for device time.
