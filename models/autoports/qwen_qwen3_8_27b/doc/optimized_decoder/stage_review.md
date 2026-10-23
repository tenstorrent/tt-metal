# Stage Review

Verdict: clean-pass

Independent review of stage03, Qwen/Qwen3.8-27B optimized single-device decoder, completed on 2026-10-23 after the stage owner declared the final artifacts ready. This review inspected the live `exp26-qwen38-bringup` worktree based on commit `22612a5fc90813932f3212bc3d8d842fa9142706`.

Reviewed optimized source SHA256: `ff85733835d1d972fb9bb1bdc47cdc761c1a814412d534681834b66d8f05d31a`.
Reviewed runner SHA256: `a991f866b4dbf2b50031431ace8c4fa37a707d8134a437da7709e0fda7c9cb73`.

## Required Work

None. The final implementation and evidence satisfy the stage contract. Review findings were addressed before this verdict. The stage owner's subsequent status updates and local checkpoint commit follow the required review-before-commit order; no push is authorized by this review.

## Other Concerns

- Small host-timing differences are not reliable extra speedups. The paired packed-output comparison consistently favored the retained layout, and the final default reproduces its decode performance. The reported final measurements, rather than the best historical samples, are used in the README and packet.
- The raw HF-cache diagnostic can fall below .995 because optimized BFP4 projections and BFP8 storage differ from the HF BF16 cache. This does not replace output acceptance: remapped optimized caches are checked for exact equivalence, and remapped outputs pass the output PCC gate.
- Historical experiments retain their actual configurations. In particular, the dominant-role geometry table is identified as the BFP4/LoFi sweep before the final packed-output refinement; its measurements are not represented as new final-default runs.

## Hard-Check Gaps

No uncovered hard check required by this stage remains. The manifest contains 17 required records: both kinds' pytest, watcher, exact maximum context, nonaligned maximum context, final profiles, B8/B16/B32 workloads, plus ragged full-attention decode. Raw request-boundary results and source inspection supplement that manifest.

The evidence does not measure dynamic trace allocation or peak request memory. The memory ledger explicitly records setup-only snapshots and leaves actual dynamic trace bytes unknown; `trace_region_size=0` is a configured reservation, not a zero-memory claim. This is a disclosed limit of the layer-stage evidence, not a serving-capacity certification.

## Anomaly Ledger

1. **Observed anomaly:** Retaining an eager L1 output caused capture to miss a cached program; exceptional cleanup could leave capture active.
   **Evidence:** `AUTODEBUG.md`, `AUTOFIX.md`, `runner_before_warmup_fix.py`, `released_warm_l0.json`, `cleanup_injected.log`, current runner.
   **Affected path:** Linear decode warmup/capture and failed-capture cleanup.
   **Control or comparison:** Releasing only the warm output restored tracked deterministic replay; injected capture failure printed `TRACE_CLEANUP_OK` and closed the device.
   **Likely subsystem:** L1-sensitive program preparation and trace lifetime. The exact original failed cache key was not logged and is not claimed proven.
   **Investigation performed:** Independent AutoDebug source inspection, discriminating lifetime change, explicit cleanup negative test, final watcher/replay validation.
   **Resolution:** Fixed. Final allocation tracking includes program-cache allocations and uses no exemptions.

2. **Observed anomaly:** Nonrectangular prefill norm grids produced enormous finite values and apparent NaN correlation statistics.
   **Evidence:** `AUTODEBUG_prefill_norm.md`, `AUTOFIX.md`, `prefill_norm_probe.json`, norm factory/reducer source.
   **Affected path:** Experimental width-sharded prefill normalization.
   **Control or comparison:** Original max magnitude about `3.0971e38`, PCC .419/.575; corrected complete rectangles max 50.25, PCC .99999842. FP64 statistics distinguish finite corruption from FP32 statistical overflow.
   **Likely subsystem:** Secondary reduction reads holes in the bounding rectangle of a nonrectangular worker set.
   **Investigation performed:** Independent source diagnosis, same-input norm probe, adapted block-sharded whole-layer controls, final boundary/watcher coverage.
   **Resolution:** Fixed locally by rectangular layouts and internal padding. No shared upstream kernel was changed.

3. **Observed anomaly:** Short/prime-row norms and B32 MLP intermediates exceeded L1; a length512 fixture left no room for its decode token.
   **Evidence:** `final_validation_round1.log`, `final_validation_round2.log`, `work_log.md`, current `_norm`, `_linear`, `_finish`, and request-capacity code.
   **Affected path:** Awkward prompt lengths, batched decode, and the validation fixture.
   **Control or comparison:** Rectangular row padding and compact `[1,B,N]` MLP intermediates pass the refreshed odd-batch boundaries and B32 workload; fixture capacity includes the next token.
   **Likely subsystem:** Tile padding, shard capacity, and test allocation arithmetic.
   **Investigation performed:** Failed-shape analysis followed by adapted layouts and current-source reruns at neighboring lengths and batch sizes.
   **Resolution:** Fixed without reducing logical-length or batch capability.

4. **Observed anomaly:** Early low-precision and multi-reader trials failed on synthetic inputs or incompatible weight padding.
   **Evidence:** `recorded_sweep.log`, `reader_geometry_sweep.log`, `all4_r*`, `g*_r*`, candidate policies/JSONs, `screen_results.csv`.
   **Affected path:** Material attention and MLP projection precision/geometry selection.
   **Control or comparison:** Real model weights with recorded HF activations pass BFP4/LoFi. Explicit bank/reader-compatible zero padding and logical-channel cropping restore valid multi-reader execution.
   **Likely subsystem:** Numerical sensitivity of the test distribution and physical shard shape, not a blanket BFP4 restriction.
   **Investigation performed:** Real-input precision/fidelity controls, compatible geometry cross-product, larger-block adaptations, and final per-role reader repeats.
   **Resolution:** Fixed/controlled. Synthetic failures did not veto the accepted real-weight winner.

5. **Observed anomaly:** Initial prefill advice rejections came from an older packed topology and did not establish the best separate-prefill path.
   **Evidence:** `review_prefill_advice.py`, `review_prefill_frontier.py`, `review_prefill_chunks.py`, `review_l1_*`, their result JSONs, and work-log comparison tables.
   **Affected path:** Prefill grid, chunk, norm layout, K blocks, and input placement.
   **Control or comparison:** Matched separate-prefill trials selected 11x10, K10/K32, linear1280/full1024 chunks with matching norm limits. Larger legal blocks and coherent L1 input were adapted and measured; slower or exact-capacity failures are recorded.
   **Likely subsystem:** Phase-specific matmul geometry and L1 allocation lifetime.
   **Investigation performed:** Review-triggered matched topology sweep, chunk neighbors, wider-grid legal-block frontier, and L1 lifetime repair before rejection.
   **Resolution:** Fixed. Winners are in the default and covered by fresh boundaries, watcher, batch, context and profile runs.

6. **Observed anomaly:** The earlier default left a small consistent packed-output decode gain and width1 prefill subblock advice unresolved.
   **Evidence:** `compare_final_decode.py`, paired result JSONs, `probe_prefill_subblocks.py`, final-advice policies/results, `before_final_advice/optimized_decoder.py`, accepted runtime attributes.
   **Affected path:** Decode gate/up output layout and prefill output/down projections.
   **Control or comparison:** Alternating paired trials favor `l1_mlp=True`; ten-column grids do not give a clear overall win. Legal width5 linear/width3 full trials pass and improve prefill.
   **Likely subsystem:** Output movement and legal non-power-of-two output subblocks.
   **Investigation performed:** Matched whole-layer controls, explicit test-only factory overrides, guarded production wiring, and a complete final revalidation after the change.
   **Resolution:** Fixed. Actual final CSV attributes contain width5/3 and two-reader BFP4/LoFi matmuls; final device prefill and decode improve over the preceding selected profiles.

7. **Observed anomaly:** Per-op perf-report FLOPs percentages exceed100% on some DRAM-sharded rows.
   **Evidence:** Accepted decode perf reports, raw matmul attributes, reader microprofiles, `report_optimized.py`, `performance.json`.
   **Affected path:** Interpretation of local profiler utilization.
   **Control or comparison:** Rows display eight storage banks while attributes select two reader/compute workers per bank. The stage roofline independently uses all120 physical ASIC cores and the complete layer envelope.
   **Likely subsystem:** Per-op core-count denominator in the reporting tool.
   **Investigation performed:** Compared displayed cores with actual reader attributes and independently recomputed whole-layer rooflines.
   **Resolution:** Controlled. No percentage is clamped; anomalous local percentages are not copied into headline utilization or telemetry.

8. **Observed anomaly:** Earlier runner hashes differed, and generated prose could describe stale policy/timing details.
   **Evidence:** `runner_metadata_delta.json`, fresh `baseline_accepted_l0/l3.json`, final validation provenance, regenerated README/results, selected-policy literal comparison.
   **Affected path:** Evidence provenance and before/after comparability.
   **Control or comparison:** The earlier pytest metadata-only delta was exactly reconstructed. Both accepted fused baselines were subsequently rerun with the exact final runner; all final source hashes match. Historical topology and setup-only memory claims are explicitly qualified.
   **Likely subsystem:** Live-worktree evidence generation during optimization.
   **Investigation performed:** SHA256 checks, source/snapshot comparison, exact-runner baseline refresh, and final documentation reconciliation.
   **Resolution:** Fixed. Current baselines and optimized runs share runner/source provenance and identical headline workload metadata.

9. **Observed anomaly:** Raw HF-cache PCC is below the output threshold for a remap test.
   **Evidence:** `pytest_contract_l3.json`, remap assertions in the runner, README cache explanation.
   **Affected path:** Comparison of quantized optimized cache values with HF BF16 values.
   **Control or comparison:** Original and remapped optimized caches are exactly equal after corresponding page ownership changes; remapped output PCC is .99989115.
   **Likely subsystem:** Deliberate projection/cache precision differences.
   **Investigation performed:** Inspected both the diagnostic field and independent exact optimized-cache/output assertions.
   **Resolution:** Controlled. The diagnostic is not treated as a passing output-PCC measurement.

10. **Observed anomaly:** Clock-settling, motherboard/subset-discovery, optional HF-kernel, Matplotlib-cache, pandas-column and Tracy-WASM-copy warnings appear in logs.
    **Evidence:** `final_validation.log`, accepted profile logs and complete raw/report CSVs, work-log warning classification.
    **Affected path:** Validation clock setup, hardware discovery, CPU reference, and report tooling.
    **Control or comparison:** The1343MHz warning is within firmware tolerance and occurs in validation; both accepted headline profile headers report1350MHz. Device0 is selected consistently. Complete device timestamps and result rows remain available; runtime conversion audits pass.
    **Likely subsystem:** Setup/reference/reporting, not an observed TT decoder execution failure.
    **Investigation performed:** Separated warnings from compiler command flags, checked profile headers/row completeness and runtime identity, and inspected reference/runtime boundaries.
    **Resolution:** Controlled. No unresolved device fault, corruption warning, or runtime Torch fallback remains in accepted evidence.

## Scope Inspected

- **Goal/skill paths:** [stage_contract.md](stage_contract.md); installed tt-model-bringup0.1.21 `stage-review`, `optimize`, `agentic-serving`, `tt-enable-tracing`, `tt-device-usage` and startup requirements; `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- **Implementation:** [optimized_decoder.py](../../tt/optimized_decoder.py), baseline functional/fused interfaces, FP32 recurrence wrapper and reader/writer lifetime rules. The optimized class is independently implemented; the fused class is selected only by the explicit test control. Construction/reference Torch use is separated from audited device forwards.
- **Harness/report code:** `run_optimized_decoder.py`, `test_optimized_decoder.py`, `ragged_optimized_decode.py`, validation/profile wrappers, fixture generation, reader microbenchmarks, sweep/adaptation scripts, collectors, roofline reporting, manifest/packet finalization and result rendering.
- **Artifacts:** README, work log, checklist, selected policy, AutoDebug/AutoFix reports, norm probe, all17 final manifest records, exact-runner baseline and accepted JSONs, raw `ops.csv`, complete-window timing JSONs, both-phase advice reports, final4096/513 screens, all48 accepted reader rows, material precision/geometry/topology/L1/subblock controls, context and agentic-serving ledgers, and the required external packet.
- **Packet:** `/home/mvasiljev/agentic-research-data/experiment-26/bringup-logs/qwen38-27b/20261023-astra-medium-exp26-clean/telemetry/packets/ee9997b0-4c77-4500-b9e6-9af11e547881.json`. Run/attempt/stage identifiers,4096+128/B1/concurrency1 workload, per-kind values and every referenced evidence path were checked.
- **Commands run:** Read-only `rg`, `sed`, `cat`, `tail`, `git status/diff`, SHA256/diff inspection, and small standard-library Python scripts using AST/JSON/CSV for independent policy, provenance, PCC, timestamp, reader, arithmetic and packet checks. The reviewer opened no device, launched no hardware tests or server, performed no reset, and edited only this report.

Final correctness checks establish current-source audited forwards, both layer kinds, bitwise restored-state replay, changed inputs/positions, randomized page ownership and live remap, resume at128, and ragged B8 positions. Request tests include all earlier1–513 boundaries and the new1279/1280/1281 plus426/427 linear boundaries and1023/1024/1025 plus341/342 full boundaries. Minimum request PCC is .99943625/.99952550. Exact262144 prefill and262142 plus two final decodes pass for both kinds; B8/B16/B32 pass4096+128. Watcher and profiler are separated.

Independently recomputed whole-layer device timings, in microseconds:

| Kind | Fused prefill | Optimized prefill | Speedup | Fused decode | Optimized decode | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_attention | 82204.061 | 16057.129 | 5.119x | 2324.989 | 815.302 | 2.852x |
| full_attention | 92975.464 | 17831.050 | 5.214x | 2188.542 | 704.049 | 3.109x |

These are `max(DEVICE FW END CYCLE) - min(DEVICE FW START CYCLE)` over the complete signposted device window, divided by1350MHz. They include every operation and internal gap. Accepted prefill contains20 material matmuls per kind; decode contains4. Every material weight row is BFP4 with LoFi, and actual decode attributes select two readers per bank. Final one/two/three-reader repeats favor two readers for every role in both kinds. Profiler-off4096 decode screens reproduce795.192/688.602us; final513 screens also improve over matched fused controls.

The headline optimized prefill/initial-decode/final128th-decode PCCs are .99980716/.99981594/.99982184 for linear and .99986445/.99959877/.99963450 for full. All exceed .995.

Rooflines were independently recomputed using one ASIC's663.552TFLOP/s LoFi theoretical peak and512GB/s DRAM peak. Useful prefill work is3,163,039,465,472 /3,257,527,418,880 FLOPs; estimated compulsory decode traffic is223,195,136 /218,601,600 bytes. Ideal prefill times are4766.830/4909.227us; ideal decode DRAM times435.928/426.956us. Complete-window utilization is29.687%/27.532% compute and53.468%/60.643% DRAM. Host times remain separately labeled. Physical padding/headers and state traffic are included in the estimate; omitted rereads/spills are disclosed.

## Residual Risk

- This is a single-layer supplied-hidden-input result. It certifies neither full-model accumulated quantization error nor generated text quality, token feedback, scheduler behavior, multi-chip scaling, vLLM integration or serving trace reuse.
- Maximum-context fixtures repeat recorded HF activations. TT computes all outputs, while the long reference compares the final128 prefill outputs using the complete prefix and the final two decode positions. This establishes tested capacity and layer semantics, not arbitrary long-conversation accuracy.
- Maximum context is validated at B1 and larger batches at4096+128; the full batch/context Cartesian product is not claimed. Setup snapshots, layer-local KV/recurrent estimates and unknown dynamic trace bytes do not establish a full-model peak-memory budget.
- The canonical default is validated. Arbitrary experimental policy dictionaries and every possible logical length are not exhaustively tested; the affected branching boundaries and awkward tails are covered, and source inspection found no public alignment restriction beyond the inherited128-aligned continuation-start contract.
- Measurements establish the tested winner on the selected device and workload. Small sample variability and untested future compiler/hardware changes remain ordinary performance risks.
