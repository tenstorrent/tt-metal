# Stage 05 work log

Baseline: bdbdc9e17c3f5dbe0ac12ac1e44bc408c81f8da0, completed TP4 decoder. Stage scope is decoder only. HF revision842da3794eaa0b77d5f08bae87a17459d91ff475. Local tt-smi lists four p300c ASICs; selected mesh1x4 Ring. Device commands serialized. Startup read run_contract.json, agentic_evals/inputs.json and installed workflow skills; environment.py validated installed AutoDebug dependency.

## Initial operation topology audit

| Boundary | Current sequence | Candidate/action | Constraints/evidence |
|---|---|---|---|
| Input attention | L1 RMSNorm, working reshard, packed QKV or tied QK, head split/norm/RoPE | Preserve packed BFP4/LoFi; compare wider working shards against final geometry | stage04 q12/q24 and qk21/3reader trials; recheck final-policy coverage |
| Attention output | local DRAM-sharded MM, slice, DRAM, RS, AG, L1 norm, residual | Compare output-sharded AG-MM decomposition; preserve prior sharded-carry families | BF16 decode/BFP8 prefill CCL; two links; persistent decode buffers |
| MLP | norm, shared working reshard, separate gate/up, fused GELU multiply, down, RS, AG, norm, residual | Packed vs separate already compared; remove redundant prefill typecasts; inspect activation and norm costs | BFP4/LoFi weights; BFP8 prefill intermediates; final geometry trials required |
| Residual carry | replicated across mesh, eight-core width-sharded L1 decode | Sharded1344 carry with distributed norms and delayed/fused AG previously measured; audit full family evidence | No layer-boundary CCL; comparison-only gather outside measured window |
| CCL lifetime | independent RS staging/output and AG output per role/B1,8,16,32 | Preserve persistence and test after changes | buffers live with decoder, overwritten before use |
| Prefill | internal1024 chunks/tails, minimal matmul, DRAM/L1 helpers | Remove identity typecast and measure norm/layout alternatives | preserve arbitrary logical lengths and262144 context |

Baseline complete-window profile: stage04 tracy_final/{sliding_attention,full_attention}/batch1/performance.json. Required4096/128/B1/concurrency1. Target agent workload estimated32 agents x32768 live tokens, typical4096; batches1/32/16/8.

No stage05 result claimed yet. Prior artifacts are evidence to inspect, not automatic stage05 signoff.

## First measurements and adapted fused decomposition

Fresh baseline commands: `TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_multichip_decoder --layer {0,5} --seq4096 --inputs bringup/artifacts/optimized_decoder_layer{0,5}_inputs262144.pt --replicated --cyclic --result <before_layerN.json>` (actual argv, including separated flags, in manifests).

The direct-row candidate removes L1-interleaved slice → L1-width-shard → DRAM, using DRAM slice directly after sharded matmul output. This removes two layout operations per row projection. Identity BFP8 prefill cast is also removed. Sliding PCC is unchanged; host warmed prefill11282.0→11131.2us, trace433.39→428.39us. Device profiles are still required; these numbers are host timing only.

Output-AG decomposition packs the real O/down weights along local output width1536 (1344 useful plus192 zero), gathers local input channels inside fused minimal matmul, slices local padding, then gathers only for the replicated residual contract. BFP4/LoFi is preserved; weights are interleaved for this API, in contrast to the selected DRAM-sharded path. The first prefill PCC=.984939 with BFP8 MLP input; decode passes. Explicit BF16 fused input restores PCC=.999921, decode=.999930 at12817.9/670.3us. A smaller K4/N4 adaptation passes but is slower14371/908us. This is a numerical contract adaptation, not rejection at the first API error.

The sharded-residual output-AG family carries1344 channels through post norms, residual adds and the next input norm. Harness gathers only outside timing. It passes prefill/decode PCC=.999916/.999908 but takes12203.7/933.2us. It therefore loses to direct-row replicated carry without charging an immediate old-contract restore. These are complete layer times. Prior fused-MM-RS/sharded and input-AG-MM/sharded alternatives remain in stage04 artifacts and are explicitly compared, not replaced by this distinct output-projection decomposition.

The prefill norm candidate initially failed a Python shape-slicing API assumption; converting the shape to tuple fixes it. No device stall or numerical rejection was involved.

## Cumulative controls

Both meaningful layer kinds were remeasured with final BFP4/LoFi roles at Q/QK sender14/K12 and7/K24, gate/up14/K12 and7/K24, down8/K22 and O4/K16(sliding)/K32(full). The controls include upstream working shards and whole-layer overhead. Full QK7/K24 improved the direct-DRAM slice screening path but loses when crossed with the stronger norm/pool/L1-slice family (478.77us versus475.04us), so it is not selected. Larger K blocks are not rejected using a different precision policy.

BFP8/HiFi2 and BFP8/LoFi controls were measured independently for attention, gate/up, and down, with all other groups held at BFP4/LoFi. All pass real-output PCC but lose traced latency. Independent BFP8 decode input-activation trials for attention and MLP also fail to beat the best cumulative family. Both layer types' combined BFP8 decode CCL/persistent-L1 candidates pass but do not improve on BF16 (sliding427.70us, full476.97us); BF16 is retained. Prefill CCL remains BFP8, with duplicate identity conversion removed. Selected runtime dtypes remain subject to final profile verification.

The complete-run baseline timeline is setup/weight upload → eager warm prefill (four1024-token chunks at4096) → decode program warmup → trace capture →128 replays. Profiling advances real recorded inputs/positions with harness copies outside each whole-layer device window. Profile readback drains occur between groups, not inside layer windows. Warmed device time includes norms, projections, attention/cache, collectives, layout operations and internal device gaps. Host replay timing and profiled input-copy overhead are separate; this layer has no token sampler or generator loop.

## AutoFix and final-path acceptance

The7-column FP32 prefill norm hit L1 scratch overlap even after in-place output adaptation. AutoDebug inspected generic norm allocation and recommended8 columns,21 width tiles and subblock3, reducing static scratch by100352 bytes/core. AutoFix also corrected occupied shard-row counting for awkward tails. `AUTODEBUG.md` and `AUTOFIX.md` preserve the diagnosis, failed adaptations and repair.

The selected cumulative path is norm8 + L1 slice + shared persistent L1 AG output, with the existing BFP4/LoFi/BFP8-KV precision policy. `check_multichip_shared_workspace` passed32 alternating replays over four live traces (two layer types × B1/B32), with allocation tracking and program-cache allocations included. Buffer addresses are identical across all logical batch views and layers, and outputs do not alias the pool. Only exact short-lived captured output tensors are acknowledged.

Final acceptance command: `python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_optimized_multichip_acceptance`. It serializes boundaries, norm_boundaries, batches, stress, watcher, stack, context and profiles, aborting on first failure. Source formatting happened during early boundary checks; `formatting_bridge.json` proves identical non-import AST for all affected sources. The production implementation is fixed for this queue. No device commands overlap.

An additional isolated-reader microprofile uses actual projection operands, the selected BFP4/LoFi geometry, one/two/three readers where bank-width divisibility permits, and alternating forward/reverse trial ordering. It is separate from the whole-layer headline roofline. `run_multichip_readers` must run only after the acceptance queue releases the mesh.

## Final measurements and supplemental advice

All54 final correctness cases passed, including maximum logical262144, both layer kinds, B1/8/16/32, non-aligned boundaries, cyclic-cache wrap and128 advancing stress replays. Minimum stress PCC is0.99986447. Two-layer integration has no boundary conversion. The final-source shared-pool rerun passed four live traces and32 alternating replays on all devices, with trace allocation tracking and program-cache allocations included. Worker watcher is clean under the documented Ethernet instrumentation limit.

Final whole-layer device measurements (prefill/decode microseconds) are in performance.md. B1 sliding11422.54/484.77→11150.01/475.22, full13748.47/534.77→13488.43/527.82. All four batch rows improve both device phases. The final default reproduces the selected candidate's traced host timings within0.06% for B1/B32. Headline numbers come from final default profiles, never from earlier candidate timing.

The first final full-attention B8 capture completed its numerical workload but failed profiler postprocessing with an unmatched SdpaDecode marker. Preserved the failed capture, checked no device-file owners and bounded tt-smi visibility, then reran only the identical failed profile: `python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_multichip_profiles --layer 5 --batches 8 --output-root models/autoports/google_gemma_4_31b_it/doc/optimized_multichip_decoder`. It passed. No reset was needed. profiler_anomaly.json and acceptance_completion.json preserve the original failure and controlled resolution.

`python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_multichip_supplemental` completed shared-workspace final-source replay, real4096-token prefill trace trials and alternating-order reader microprofiles, serially. Two readers wins every role; a legal three-reader full-QK variant loses. Prefill trace modestly improves sliding and does not improve full attention, with capture overhead recorded. Details are in profile_analysis.md.

Every larger-batch input-L1 recommendation was reviewed. Commands use `python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_optimized_multichip_candidates --layer {0,5} --batch {8,16,32} --names <case>`; exact argv/source hashes and exit codes are in each artifact's manifest/exit JSON. Q/MLP B8 needed M1/K8/N8 after static-buffer overlap, down B16 needed M1/K4/N4, and O controls were retried at smaller blocks. The full-O B16 warmed128MiB allocation probe fails on contiguous space, not totalfree capacity. Larger-role input-only bounds exceed total L1 capacity. Two internal-chunk families at1024 and2048 physical rows remove those limits and are measured across both kinds and all larger batches. All correct adapted families are slower. input_l1_trials.md and compact capacity/failure JSONs preserve measurements and exact blockers; no first API error alone rejects an optimization.

Production precision and context remain unchanged. The shared workspace reduces layer setup DRAM overhead while adding688128 pooled L1 bytes per device independent of layer count. memory_plan.json and doc/context_contract.json carry the physical allocation accounting and inter-layer contract. All raw profiler captures, tensor fixtures and ignored logs remain local; only source and compact evidence are checkpointed. No full-model or vLLM work, push, issue comment or external message was performed.

Final host checks: repository pre-commit over all stage-owned modified/new files passes (initial pass only removed trailing whitespace from generated text tables); compileall for tests/tt and git diff --check pass. Production SHA256 remains7c21e0d36423a4305bd8b07cf8a805909b3145168ecba3703500cdabb9304837, matching final shared-workspace replay. No local device process remains from this stage. The required compact packet is written at bringup/artifacts/multigoal-runs/20261030T163027Z/continuation-02-20261030T182350Z/telemetry/packets/a40cd227-5628-4d84-b89b-edd36b6c8bb0.json; its summary script starts from the supplied template and includes only matching4096/128/B1 whole-layer device values.

Independent stage review: stage_review.md, verdict clean-pass with no required work. Reviewer checked final production hash, all54 cases, eight profiles, whole-layer roofline arithmetic, all remaining L1 trials and required telemetry. Parent read the final report and verified cited compact artifacts. Context/agentic status and telemetry were refreshed to validated after the verdict. Only tt-metal has stage-owned changes; no other repository needs a checkpoint.
