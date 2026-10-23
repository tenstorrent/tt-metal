# Stage04 work log

2026-10-23. Initial checkout7c620da56e3475f64e8656d70cde0a9b6464bb20; clean worktree. Enabled AutoDebug/model-bringup/telemetry confirmed in host inventory/config; package environment.py validated. Selected skills: multichip, agentic-serving, tt-device-usage; tt-enable-tracing used for capture validation. Scope remains decoder/tests/docs.

Hardware commands run serially in `qwen38-functional-build`, `/workspace` mapped to checkout. Use `USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache PYTHONPATH=/workspace`. Correctness adds `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`. No watcher/profiler overlap.

- `timeout 60 tt-smi -ls --local`: four p300c ASICs. Snapshot in hardware_snapshot.json.
- Initial raw2x2 FABRIC_2D smoke failed before kernels because TT_METAL_CACHE was omitted (permission denied under `//.cache`). Corrected environment. Bounded `timeout 180 tt-smi -r` returned0; list_after_reset.log shows all four chips.
- `timeout 90 python tests/probe_multichip_mesh.py --descriptor doc/multichip_decoder/cluster.yaml`: actual command paths have autoport prefix.2x2 FABRIC_2D open/close passes,2 links. `--shape 1 4 --fabric FABRIC_1D_RING --descriptor .../cluster_ring.yaml` also passes. Topology has connected edges0-1,1-2,2-3,3-0, two channels each. ETH_LIVE_STATUS0 in SMI snapshot alone was not a connectivity diagnosis; UMD discovery and open prove topology.
- Plan recorded in mesh_plan.md before writing final-path candidate. Selected1x4 TP4. Context target262144 unchanged; memory contract adds planned, explicitly unvalidated multichip entry.
- First run `timeout 240 python -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder --layer 0 --output .../smoke_l0.json`: new harness first compiled input copies after capture. Allocation tracker rejected replay. Fixed by preallocating step tensors and warming copies before capture. No exemption. Second run exposed None vs[] physical_device_ids argument before multichip open; corrected.
- smoke_l0_v3.log:33-token prefill +2 advancing traced decodes, B1, recorded HF activations, real weights, direct optimized-TTNN control. PCC prefill0.9999550974, decode0.9999961276/0.9999954554. Host last-step795.58us baseline/514.83us TP4; diagnostic only, not headline/performance acceptance. Tracking remains enabled; no host torch or conversion inside forwards.
- smoke_l3.log: full-attention prefill ran; decode exposed stale6144 global reshape for local1536 output. Fixed model-local reshape. Bounded all-device reset recorded in reset_l3.log before retry. This is an implementation shape error, not a hardware defect.

Status: implementation/validation ongoing. No pass, checkpoint commit, full-model or serving claim.

- Full-attention reshape repair passes in smoke_l3_v2.log:33+2 PCC0.99997808/0.99999816/0.99999922.
- Hidden-sharded residual candidate initially hit distributed norm's rank4 input requirement. Adapted both pre/post norm to explicit4D views while preserving public3D layout; sharded_l0_v2.log passes (PCC0.99995815 and>0.999993 decode). It consumes sharded residuals directly through add and distributed stats; the test boundary gather is excluded from layer timing.
- workload_l0/l3.json:4096+128 B1 comparisons pass, all output replicas equal, warmed prefill bitwise equal,3 restored decode replays bitwise, host-conversion audit clean. Full attention remaps physical pages and table under a live trace and reproduces output exactly. Current memory/profile numbers in performance.json are candidate results.
- Fused-both experiment stalled in first AG+mix. Captured tt-triage before terminating only this run's PID2620547. Bounded all-device reset returned0; list_fused_reset.log shows all4 devices. AUTODEBUG.md diagnoses indexed-worker list ordering across a core row. AutoFix reduces fused AG worker count2→1 with2 links intact; fused_both_w1_l0.json passes eager, trace and PCC but is slower. No common-source patch. AUTOFIX.md records follow-up.
- profile_l0 and profile_l3 run4096+128, selecting last valid decode step and warmed prefill trace. Watcher disabled. Whole-layer device windows530.54/454.01us decode and9251.55/11460.01us prefill. Raw captures retained locally, compact human tables and CSVs under tracy/<label>/. tt-perf-report estimates8 matmul cores; actual runtime attributes explicitly request2 readers/bank, and total CORE COUNT includes110 storage/worker cores. Do not use either display field as whole-ASIC peak. Whole-layer rooflines use4×120 physical cores and4×512GB/s.

### Final candidate selection (19:20 UTC)

`phase_mix40_l0.json` and `phase_mix40_l3.json` validate the coherent phase-specific residual family on 4096+128, batch 1: hidden-sharded prefill and replicated decode. They retain the same state layout, weights, BF16 activations/CCL and BFP4/LoFi projections. Decode uses 40 input cores and K block 4 for the mixed projection; other selected projection geometries remain unchanged. The default constructor and test CLI now select this family. The phase change is static at setup, with separate method dispatch; there is no data-dependent runtime fallback. Each phase is stack-compatible without intermediate host conversion.

Batch, boundary, native-context, continuation, stack and watcher validation follow this selection. Earlier replicated-path profiles remain comparison evidence, not final selected measurements.

Harness provenance after selection: batch runs share the selected decoder implementation hash. During the serial batch suite the harness gained optional split-prefill continuation coverage, a local-state PCC gate, and cleaner RoPE slicing syntax. These additive changes do not change ordinary batch input/forward execution. The final profiles and later validation record their exact runner hashes. Compact measured-window operation CSVs and tt-perf-report CSVs explicitly override the repository's general CSV ignore; full-run raw CSV/captures remain local.

### Reverse-stack control trace preparation

`stack_l3_first_failure.log` records a single-chip optimized control failure: after eager decode, capture tried to load a new uncached program. The harness retained the eager output, whereas the validated optimized-stage harness deletes it before capture. Optimized decode can return L1-sharded output; retaining it changes L1 allocation conditions for the next full→linear stack. The selected TP4 interface returns DRAM output. Saved live stacks in `triage_stack_l3/`, terminated only the failed process, then performed bounded reset (`reset_after_stack.log`) and successful local device listing (`health_after_stack.log`). Minimal test-harness correction: delete the consumed eager output before capture. No trace exemptions or selected implementation change. `stack_l3_lifetime` is the discriminating rerun.

### Watcher instrumented fabric size

Both native-context fixtures passed, including all 262,142 prefill outputs and decodes at positions 262,142/262,143. The first watcher run failed at TP4 mesh open, before model construction: fabric router binary 29,264 B exceeded ACTIVE_ETH's 26,624 B kernel config buffer. `watcher_l0_first_failure.log` preserves the exact size breakdown. The documented `TT_METAL_WATCHER_NOINLINE=1` option (`tt_metal/llrt/rtoptions.cpp`) outlines watcher functions to reduce binary size; it does not disable checks. Add it only to the separate watcher suite, then reset and retry. No model or shared TTNN source change.

### Validation gates complete

`python3 models/autoports/qwen_qwen3_8_27b/tests/summarize_multichip.py` passes: batch 6 cases, boundary 25, continuation 4, mixed stack 2, native context 2, watcher 4 (43 cases). `validation_summary.json` checks each selected implementation hash, finite PCC >=0.995 for every output/state comparison, fallback audits and three exact restored replays. The reviewer identified a potential NaN weakness in `min(scores)`; explicit finite checks now cover existing and future evidence. This changes only result validation, not executed kernels.

Watcher uses `TT_METAL_WATCHER=100ms TT_METAL_WATCHER_NOINLINE=1`, all features enabled. Both 4096+128 B1 layer workloads and both B32 sequence-33 workloads pass. Per-case `*.watcher.txt` snapshots contain no error/assert/corruption report. Native context tests compare all prefill outputs and both final decodes, not a shortened prefix or subset.

Final fused alternatives: `alternatives_validation.json` records four coherent 4096+128 cases (RS-only and combined 1D AG+RS, both layer kinds), all PCC/state/trace checks passing but decode slower. Combined medians681.52/599.03us, RS-only690.96/600.34us versus selected490.06/408.52us. Three-reader outlier control median521.61us over128steps, range518.01–536.42us; prior1029us last-step spike not reproduced. Final 40-core/K4 three-reader candidate487.99us overlaps the selected two-reader timing, without a material win; retain the common two-reader packing.

### Packed-prefill selection and memory refinement

Review requested a TP4-specific packed-prefill control. `packed_prefill_l0/l3.json` passes; repeated controls in `packed_validation.json` use ten restored warmed prefill replays and128 advancing decodes. Packed/separate medians8481.95/8557.71us linear,10786.09/10915.79us full. Select packed for both phases and omit unused separate gate/up allocations. Measured savings26,542,080 bytes/device/layer,1,698,693,120 over64 layers. The old43-case validation is preserved under `before_packed/`; `final_validation.log` reruns all required cases and coherent fused alternatives against the new default. No previous-stage source modified.

Reviewer found that cached watcher jobs could finish before the old10-second interval. Reran only the four watcher cases with100ms polling. `watcher_scan_summary.json` records53/59/53/55 completed scans, with real post-startup kernels and no error/assert/corruption indications; full copied watcher logs are retained. No model-code change.

### Profiler scratch capacity

Before final profiling, root filesystem had only4.9GB free; the earlier linear raw capture alone contains6.15GiB before report copies. Preserved the generated kernel cache using `tar -I 'zstd -T4 -3' -cf /data/mvasiljev/autoport-cache-backups/qwen38-stage04-kernel-cache-20261023.tar.zst -C .cache tt-metal-cache`. Archive creation returned0; `zstd -t` verified25,951,283,200 uncompressed bytes. Cleared only `.cache/tt-metal-cache` (rebuildable compiled kernels), leaving source, model weights and all evidence intact; free space became30GB. Cache can be restored with `tar --zstd -xf <archive> -C .cache` while hardware jobs are stopped. Final profile cold compilation occurs before warmed capture/replay and is outside the measured windows.

### Final prefill geometry gate

After inspecting SLOW matmul rows, measured TP4 BFP4/LoFi prefill at4096tokens with five warmed replays: control, hiddenK20, mixedK40, outK24, downK34 and11×8/8×8 smaller grids on both layer kinds (`prefill_geometry_results.json`). All candidates pass PCC/state/trace checks. Linear control8290.84us is fastest; smaller grids8764.99/9448.56us. Full hiddenK20 initially appeared faster over five samples, so repeated control/K20/K20+mixedK40 with20 warmed prefill replays and128 advancing decodes. True medians10707.73/10725.69/10731.59us do not reproduce an improvement; retain current defaults and lower L1 requirements. `prefill_selection.json` contains distributions and exact artifacts. Selected source is unchanged, so final43-case validation and device profiles remain current.

Final device profiles: `final_profile_l0.log`, `final_profile_l3.log`, `tracy/final_profile_l0/` and `tracy/final_profile_l3/`. `python3 .../tests/report_multichip.py` and `python3 .../tests/analyze_multichip_perf.py` regenerate performance.json and profile_audit.json/profile_analysis.md. Linear device prefill8326.151us,decode508.853us; full10628.844us,424.104us. Whole-layer speedups1.9285/1.6022 and1.6776/1.6601. Projection runtime audit confirms BFP4/LoFi and packed8704 prefill; actual AG/RS costs about2.707ms prefill and65us decode. Local distributed RMSNorm is classified as compute, not CCL.

### Independent acceptance

Fresh xhigh reviewer `/root/stage_review` returned **clean-pass**, no required work; full report `stage_review.md`. It independently verified all43 final-source cases, finite output/state PCC, complete watcher scans, both per-device timing envelopes, runtime BFP4/LoFi/packed projection rows, speedups/efficiency/rooflines and memory arithmetic. Minimum output PCC0.9999053588, statePCC0.9998227069. Final source SHA2564b8b89ab05452f6c9194049ed6baa155265a3d1d1288d7043f8fe25c731eca90. Host syntax checks and staged source/Markdown whitespace checks pass. Generated CSV/log/text evidence retains verbatim line endings/table padding via stage-local attributes. Only decoder/tests/docs in this autoport are staged; no raw profiler captures or unrelated files.

### Local checkpoint

Repository `/home/mvasiljev/tt-metal`, branch `exp26-qwen38-bringup`: implementation/evidence checkpoint `81697dba9b6a141b0d5df41416e2b78d81dda506`. Committed after independent clean-pass; never pushed. Git had no configured identity, so the command used the existing pipeline identity `Codex <codex@localhost>` via command-local settings, without changing global configuration. A following documentation-only commit records this SHA; its own final SHA is recorded in the external telemetry packet.
