# AutoFix — initial sampling capture

Failing contract: every capture is ended before teardown; sampled token feedback
must execute a real common-sampler trace. Observations and source diagnosis are
in AUTOTRIAGE.md.

Prioritized repairs: (1) supply required grammar_on; (2) exception-safe end;
(3) print original exceptions before cleanup. All three are model-local changes
in tt/generator.py and tests/run_full_model_probe.py. No runtime/decoder changes.
Discriminating check: reduced_generate probe must leave cleanup and replay three
decode steps successfully, followed by allocation tracking. Validation pending
original bounded timeout and hardware recovery. No passing repair claimed yet.

Repair validation: reduced_generate_params.log exits0 in21s after supplying
32-element default penalty parameter arrays (common helper rejects scalars).
Two captures, three model replays, four sampling replays; tokens
[28433,150929,11927,66275]. Clean shutdown. Tracking-enabled rerun also exits0 in21s,
with TT_METAL_TRACE_ALLOC_TRACKING=1 and SKIP_PROGRAM_CACHE=0, no exemptions.
This repairs the identified capture lifecycle failure. Positive-control and
full-model correctness evidence are still outstanding, not inferred from smoke.

## Full-stack L1 allocation

Diagnosis: AUTODEBUG_L1.md identifies duplicated immutable TP4 SiLU tables; the failed 958464-byte allocation needs 29952 bytes per core but the largest free interval was18432 bytes. Layers0–28 loaded; layer29 failed before caches/terminal weights. First hypothesis: immediate model-owned sharing of the exact selected helper table removes depth-dependent L1 growth. Broader constructor injection and other constant sharing are deferred unless the narrow repair fails. `tt/model.py` validates physical tensor metadata and aliases only table storage after each completed layer; helper objects, weight-dependent norm/decay tables and all mutable state stay distinct. L1 census added alongside DRAM. Reduced0,1,3 baseline captured in three_layer_before_sharing.json; repaired comparison and full-stack validation pending.

Sharing validation: three_layer_shared.json matches all4 baseline tokens exactly and passes allocation tracking. Full40 constructor passes (logs/full_probe_shared.log); layer39 L1 residency962496 bytes/core and DRAM18453479424 bytes/chip before terminal tensors. Prefill then fails gather SiLU CB collision: residentL1 starts541888, staticCBends637248. This is execution workspace headroom, not a modelcontext limit. Narrow second intervention shares identical read-only router_bias/router_indices aftereachlayer (39*4096=159744 bytes/core saved); mutable router outputs remain distinct. Reduced probe now accepts --resident-l1-bytes to emulate fullstack pressure without reloading40layer weights. It does not reconstruct exact allocator holehistory.

Residency probe802752 fails later in fullattention SDPA: CBend887936 > L1bufferstart687808 (reduced_l1_residency.log). Routerinput sharing alone is insufficient. Source reachability: selected MultichipDecoder._moe exclusively uses MeshFP32TopK/native topk and never reads inherited router_bias/router_indices/router_out/router_out_ids/inline_selector. These unused legacy allocations are now released immediately perlayer; selected mutable router scratch remains untouched. Expectedfullstack residency572416 bytes/core. This removes dead storage rather than changing compute/sharding/dtypes. Reducedpressure check pending.

Final L1 repair evidence: reduced_l1_live_only.json passes at emulated residency572416 bytes/core and matches the baseline tokens exactly. Full40 run reaches571456 bytes/core and passes prefill top1=.98, top5=1, top100=1 (prefill_accuracy.json and logs/prefill_accuracy.log). Thus the constructor and selected prefill workspace failures are repaired without a precision, layout, kernel-strategy, or context reduction. Later fixed-slot and universal-prefill integration still requires final full-stack validation.

## Batch32 attention workspace lifetime

Reduced real layers0/3 with32slots/context1024 failed SDPA CB validation: L1 allocation879296 overlaps CBend888000. Source inspection shows `decode_batch` retained all sharded per-row head outputs until its post-loop DRAM conversion; the full wrapper adds other resident tensors. Moved the identical DRAM conversion immediately after each row, preserving head math, collective grouping, output projection, dtypes and residual layout. Discriminator: identical reduced32slot command passes in32s with100% JIT hits after change (reduced_batch32_drain.json, logs/reduced_batch32_drain.log); prior failure logs/reduced_batch32.log. No reset or capability reduction.

## Singleton terminal consistency

Paired bounded-versus-legacy probes: all10MoEtails are bit-exact, all9cases' residual/cache states are bit-exact, eight nonsingleton full-logit cases are bit-exact. Singleton logits PCC=.999987009 (belowprobe .99999), samegreedytoken. Hypothesis: decoder singleton enters terminal in L1-sharded layout whereas transported singleton is interleavedDRAM, selecting different RMS reduction. Canonicalize only terminal input toDRAM; preserveinterlayer layout. Targeted singleton exact rerun PASS (29s): prefill_equivalence/singleton_fixed.json and logs/prefill_equivalence_singleton.log. Original evidence prefill_equivalence/result.json retained as FAIL.

## Fixed-slot batch independence

AutoDebug isolated inspection (`AUTODEBUG_MIXED.md`, launcher log
`logs/autodebug_mixed.log`) identified launch-global stochastic FPU rounding in
grouped MoE as a candidate. Device diagnostic taps confirmed equal first-layer
MoE inputs/shared expert results but differing routed outputs when other slots
became active. `generator_replay_probe/mixed_moe.slots_*.pt` preserves intermediates.
The SR-off-only control makes fixed-B3 slots1/2 exact at every logged layer/logit
and all five token steps; it leaves a distinct slot0 failure because that solo
request used the B1 sparse path (`mixed_sr_off.json` plus .pt files).

Repair: added backward-compatible `stochastic_rounding=True` to the fused op
API and program-cache key. Fixed-slot batched decode passes False; prefill keeps
True and B1 retains its original sparse expert path. Tensor dtypes, math
fidelity, TP4/EP4 grouping and inter-layer layout stay unchanged. Capacity>1 now
uses one fixed-shape model trace even when only slot0 is active; capacity1 keeps
the optimized B1 trace. The same generator no longer changes numerical paths
according to which physical slot survives. Final batch accuracy, performance and
replay checks remain required before accepting this repair.

Build: `cmake --build build_Release --target ttnn --parallel 8` passed in21s.
Required runtime installation: `cmake --install build_Release --component tar`
and `--component tt_pybinds`; both passed. The first post-build run encountered
the old source-tree Python binding and exited cleanly; no hardware reset. Logs:
`build_sampling_rounding.log`, `install_rounding_runtime.log`,
`generator_mixed_fixed.log` (old-binding failure), `generator_mixed_installed.log`
(targeted repaired run).

Repaired fixed-slot runtime acceptance: targeted mixed probe passes36s (`mixed_fixed.json`), then the complete reduced replay suite passes43s (`generator_replay_probe/final.json`): all length/continuation/repeat, mixed/page-change/inactive and per-row sampling checks,32 program-cache checkpoints at8489 entries and zero growth. Actual serving command probe passes33s with two retained traces and async/state-remap controls. These establish request isolation and lifecycle; full-stack batched accuracy remains separately required.

Packaged discovery repair: the readiness runner imports generator.py as a standalone module. Relative helper imports failed before the first inference after all40 weights loaded. All generator helper imports now use absolute package paths. `run_readiness_factory_probe` exercises the exact packaged loader with real reduced layers, matches all4 baseline tokens and slot release, and passes30s (`readiness_factory_probe.json`, `logs/readiness_factory.log`). The all40 teacher retry is separate final evidence.
