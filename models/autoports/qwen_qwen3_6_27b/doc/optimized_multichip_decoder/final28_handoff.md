# Final28 handoff at the user-directed optimization-search cap

CPU-only inventory; no new device work authorized or attempted here. These items remain **untried in the named complete-family configuration**, not rejected and not required to be silently resumed after the cap. Existing gates are unchanged. Selection and the unresolved B1 prefill gate are recorded in `final28_selection.md`.

## Queue stopped before these experiments

At this snapshot, none of the named result JSONs exists for the following manifests (manifest/proposal/CPU JSONs are not result receipts):

| Prepared work | Jobs | Remaining hypothesis / qualification |
|---|---:|---|
| `b1_diagnostic27.json` |4|Same-current-harness originalTP4/candidate comparison. Existing candidate entries use minimal norm; update to final norm-off for validation. Four diagnostic jobs do not satisfy independent repeated-pair acceptance.|
| `combine_window27.json` |8|GDN disjoint combine-output windows avoid large temporal concatenation; quality, complete-layer cost and watcher scripts remain unrun.|
| `long_prefill_geometry27.json` |31|Large-row minimal/MMRS/output-block geometry, with17 short regression jobs. S33 has M1056 (not1024+32); S513 has M16384 bulk plus M32 tail.|
| `long_movement28.json` |28|Coherent gather-first/SFPU prefill normalization with `flat_gather_prefill=true`, plus MMRS placement and explicitly separate unflattened diagnostics. Do not reject flat family using unflattened timing.|
| `direct_qkv27.json` |25|Original direct-prefill QKV whole-layer comparison. Standalone non-aligned movement is already slower, so whole-layer integration is not selected.|
| `direct_qkv28.json` |7|Aligned/scoped direct-QKV whole-layer adaptation; remains unmeasured.|
| `prefill_swiglu27.json` |18|Bulk K/block/grid variation and independent M32 tail packed-SwiGLU dispatch. Neither is final-selected.|
| `attention_gate25.json` |24|Direct canonical L1 gating with both QK families in complete decoder. Existing focused native/direct B1 26.655/17.753us and B32 20.147/26.759us argue against extrapolating one batch. Two-row-band adaptation is already measured/rejected.|
| `joint_output23.json` |15|Phase-specific BF16 primary/correction joint-output policy. Isolated correctness passed; full-layer numerical/performance composition unrun.|
| `prediction_whole25.json` |30|Prediction-register retention plus static column partition composition. Isolated prediction exact/small gain exists; combined ownership and complete-layer gates unrun.|
| `long25.json` |12|S4097 complete-layer/context-shape coverage; no result receipts.|

`qkv_movement28.patch` remains an unapplied separate non-aligned adaptation (per-work-item batched tile I/O and a distinct face-fragment family). CPU ownership/source proofs do not establish correctness/speed. Original direct110 costs B1/S513306.428us versus native43.397us, B32/S51313211.941us versus2683.947us; this does not reject the unimplemented movement adaptations.

`live_row27_quality.sh`/`live_row27_perf.sh` and its unapplied composition patch remain pending. The B1 smoke passes after a host-oracle scalar-slot mapping correction, but all B1/3/17/32, state dtype, zero/small/large/poison, changed-state/rebound/queued and timing families have no broad receipts. Do not select live-row from smoke alone.

`combined26.sh` still has prediction×column1/2 quality/perf, then `column_remaining23.sh` with B14–32 residual coverage. No `prediction_column{1,2}_{quality,perf}24.json` or `column_runtime_remaining23_{1,2,3}.json` receipts exist. Existing column1 whole-layer numerical/performance evidence remains usable; these unrun stress extensions must be labeled honestly.

## Resource and native arithmetic follow-up

- Minimal norm64-owner B1 fails L1 allocation in `norm_resources26_minimal_b1.json`; the remaining minimalB32/DiTB1/DiTB32 resource jobs did not execute. Existing two-owner all-B metadata/trace tests pass. Keep norm off. AutoDebug is investigating bounded output/gamma/canonical buffer ownership; no unavoidable context reduction follows from this prototype footprint. Any repaired resource family needs real64-owner gates before selection.
- Native recurrence precise-readout/seed work already has meaningful rejection evidence: direct-FP32-seed BF16 passes sampled fixed702 quality but costs98.741us versus custom52.705us; FP32 additionally fails accumulated-update quality. A full FP32 native update rewrite remains unimplemented, not rejected. Do not repeat the already measured scalar/column or head-major readout families without a new coherent hypothesis.
- Fused joint combine/convolution FP32-L1 full/rest-only families were measured and rejected (4.23–5.13x slower complete boundary). BF16/DRAM inputs remain unmeasured, not rejected.

## Acceptance work is separate from optional search

The cap ends new tuning. It does not manufacture final-default measurements, alter .995 gates, satisfy the original roofline/device-time condition, or establish a stage-review verdict. Parent must record any user-authorized scope change explicitly. Final validation still needs default-path PCC/perf/profiles, same-harness B1 paired nonregression, selected-path long stress/64-owner resources/context and nonaligned continuation, current-build/source provenance, runtime-sentinel attachment, inter-layer boundary evidence, independent review and stage-owned local commits. Prepared68-job receipts are not completed evidence. If validation cannot be finished under the cap, record outstanding gates plainly rather than mark historical candidates final.

No production, test, main README or work-log files were edited for this documentation task.
