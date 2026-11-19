# Coherent TP4 fused distributed hidden RMSNorm family (source preparation)

Status: **CPU/source preparation only; no hardware, build, TTNN import, or production/default change by this agent.** The new helper/probe are `tt/fused_distributed_norm_candidate.py` and `tests/probe_fused_distributed_norm.py`. The earlier bounded performance audit is `remaining_bottleneck_audit24.md`.

## Why this is a distinct remaining family

The current gather-first hidden norm uses one full-activation all-gather, local hidden reduction/affine, and the conversion into the actual projection's L1 shards. The previously measured distributed composite norm uses pre-reduce, stats all-gather, post-reduce/affine and activation all-gather. Neither is a measurement of native fused distributed stats-norm plus activation all-gather. The new boundary retains the inter-layer `[1,B,1280]` BF16 hidden shard; it does not restore a replicated residual. Full hidden activations exist only as the within-layer column-parallel projection input. No inter-layer collective is added.

The probe measures both complete norm-to-canonical-input and norm-to-real-projection boundaries. Each includes two consecutive calls per trace to preserve native ping-pong state; reported time is elapsed / 2. Both candidates include the required activation gather. All input packing/layout conversions and canonical output copies are included. Projection timing also includes a stable projected output copy. The host upload, setup synchronization and result reads are excluded. Seven synchronized samples each are collected for eager and traced operation only without watcher, profiler or allocation tracking. These are isolated boundary results, not a substitute for the stage's warmed full-layer before/after numbers.

## DiT native admissibility and exact adapter

Sources: `ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/{dit_fused_distributed_rmsnorm_nanobind.cpp,device/dit_fused_distributed_rmsnorm_device_operation.cpp,device/dit_fused_distributed_rmsnorm_program_factory.cpp}`. Same-repo real model tests: `models/tt_dit/tests/unit/test_distributed_rmsnorm_fused.py` (especially `_call_op`, `_make_pob`, and the ping-pong correctness test).

* TP4 uses actual mesh `(1,4)`, cluster axis1 and Linear fabric. Never use `cluster_axis=None` or per-head norm to avoid communication.
* Input is BF16 TILE DRAM `[1,1,B,1280]`; physical rows32, logical B1..32. The shared residual remains `[1,B,1280]`. B3/17 are valid public shapes; internal tiling owns padding.
* Global width5120 is represented by four distinct rank-local1280 shards. `num_heads_per_device=1`, `per_head_norm=False` normalize the global hidden width rather than a local/head width. No bias or RoPE transformation is requested.
* Weight is actual pinned model `(1 + state_dict[norm_key].float())`, **FP32** tiled `[1,1,1,1280]` per rank. There is no BF16 gamma substitution. FP32 accumulation uses the model's HiFi4 compute configuration. Output and activation CCL remain BF16; candidate policy with lower CCL dtype is rejected explicitly by the bounded helper rather than silently changed.
* The function named `persistent_output_buffer` owns **FP32 row-major DRAM stats scratch**, not the normalized activation. The native stats-buffer factory supplies its exact shape/page geometry; row_capacity32 is passed consistently to both factory and operation. All scratch is allocated before any capture.
* Same-repo native tests establish the terminal semaphore reset/peer-increment race if one scratch/semaphore set is reused on successive calls. Each candidate owner has two independent native stats+semaphore sets and invokes phase0 then phase1. A separate CCL manager owns the activation gather's two semaphore/barrier/output sets. Setup synchronizes after all mesh semaphore allocations, before the first fabric launch. No inter-rank fence is inserted between queued calls.
* The activation gather uses the model's current DRAM/L1 placement and link count, preserves BF16, and writes preallocated full-hidden outputs. Canonical input layout comes from the actual projection entry, including `packed_gdn_correction_joint` when the packed correction family is installed. The current model supplies projection arithmetic without any source/default edits.

Memory overhead is bounded setup-owned scratch, two activation gather outputs, two gamma/input owners, and stable canonical/projection slots. At B32, each BF16 full-hidden physical tensor is327680 bytes/rank, each local1280 activation is81920 bytes/rank. The report records actual addresses and native stats factory allocations, but this prototype does not yet implement physical allocator byte accounting. No KV cache/context contract changes occur. The isolated probe's duplicate rebound owner is test-only overhead, not a proposed production allocation contract.

## Quality, rebinding and trace ownership

The control invokes the exact explicit-policy `model._norm` path, then the same canonical preparation. The initial fixture can be synthetic seed17 BF16 residuals with real pinned weights, or an explicit actual hidden-residual `.pt` fixture (`--input-pt`, `--input-key`). The report labels which was used; selecting `--norm post` without a fixture is not represented as a real post-attention residual.

Both original and changed nonuniform input banks are evaluated against independent FP64 global RMSNorm with the actual FP32 gamma. Every user on every rank and both ping-pong phases must reach PCC>=.995. Both canonical norm and complete real-weight projection contribution are compared with the current control at the same per-user/rank threshold. All quality rows are saved before raising, so a failed control/candidate does not hide the other measurements. These projection checks compare TTNN paths; they do not claim a second independent FP64 projection oracle or full-decoder/cache continuity proof.

A second DiT owner changes input/gamma/stats/semaphore/output addresses with unchanged logical geometry before traces are captured. The two simultaneously live input owners always hold opposite changed-input banks. A setup-only check poisons the first gamma owner with zero, binds that owner, then invokes the rebound owner with the actual FP32 gamma; its output must remain exact to its own eager oracle and the first owner's output/stats must remain untouched. The actual gamma is restored before any capture. This makes stale input/gamma bindings observable while preserving all measured arithmetic. Inputs, gamma, stats, all outputs and both owners are created before the first capture. Every exact eager/copy/projection-free branch is warmed; program-cache misses are then prohibited. Six traces capture control/DiT/rebound × norm-only/projected. Temporaries are released inside each invocation; only pre-capture output slots survive. There are no allocation acknowledgements/exemptions or corruptible scopes.

Tracking uses `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, verifies every trace before replay, and requires a live post-capture allocation positive control to fail with its exact buffer id. After releasing the positive control all traces must pass. Four changed-input replays per signature require bit-exact equality to that path's own eager result. Fifty changed-input trace pairs are enqueued without an inter-call host synchronization; the final outputs are verified (the prototype does not retain all100 intermediate outputs). Final eager interleaving is followed by traced projected replays and another allocation check. Instrumented timing is disabled.

## Minimal native family: repair pending, not performance-rejected

Sources: `rms_allgather/device/rms_allgather_device_operation.cpp` validates BF16 width-sharded L1 input, exact physicalM32, row-major gamma, FP32 stats with FP32 accumulation, and output width-sharded L1. FP32 gamma is supported, preserving actual Qwen affine arithmetic. The bounded input adaptation is8x1 cores × shard(32,160), block_h1, block_w5, subblock_w1, row-major orientation; all1280 channels are covered. Gamma row-major40×32 preserves FP32. Stats must include four FP32 tiles/rank because the factory derives global device divisor from padded stats width/32. Canonical one-core stats are shape32×128, shard32×128 at(0,0); stats core coincides with the first native reducer. The `persistent_output_tensor` parameter is the final normalized output (confirmed in `create_output_tensors`); its binding doc's intermediate-buffer wording is misleading.

Current factory lines around892–897 create an invalid host iterator range:

```
cores_per_device = stats_cores_vec.size() + ring_size - (1 / ring_size);
start = stats_cores_vec.size() / ring_size * device_index;
end = start + cores_per_device;
vector(begin + start, begin + end);
```

For TP4 and C stats cores, count=C+4, so every rank's end exceeds C. No legal padding/layout fixes this host range. This is a source-proven bug, not an observed device failure and not justification to reject the optimization family. The helper has no executable minimal arm until canonical native repair/build.

A merely parenthesized ceil count is insufficient as a generic repair: stats belong to a **global gathered tile stream**, not necessarily disjoint rank core subsets. For one tile/rank, tile t=rank; with P tiles/core, the destination is core-index floor(t/P), tile-offset t%P. Examples at TP4: C1/P4 → cores[0,0,0,0], offsets[0,1,2,3]; C2/P2 → cores[0,0,1,1], offsets[0,1,0,1]; C4/P1 → cores[0,1,2,3], offsets[0,0,0,0]. Capacity C×P<4 must be rejected. Writer source currently indexes stats subset using input tile id without its rank offset, so both core and page selection must be checked together. Core coverage alone does not prove that the post-reduce compute's globally bound stats CB can consume multi-core stats; a physical one-core four-tile layout avoids assuming an unimplemented reshard/gather. `num_links=1` is the bounded initial setting because the factory emits all-gather runtime work only for worker i=0.

An actual installed AutoDebug fresh-process investigation is running in `debug_fused_rms_minimal/AUTODEBUG.md`, launched with the explicitly authorized existing `AUTODEBUG_SKIP_CHILD_SANDBOX=1` after the environment's earlier sandbox preflight failure. It is source-only and independently checks writer/consumer mapping, semaphore/drain ownership, supported links and additional blockers. Parent owns subsequent AutoFix/build/hardware; no native source was edited by this agent.

## CPU validation and exact initial commands

Executed without importing Torch/TTNN or touching devices: `python3 -m py_compile` for both new files; `--cpu-only` CLI plans at B1/layer0 and B32/layer3 using explicit existing fusions20 policies; all passed. The CPU geometry helper admits every B1..32 and rejects0/33. Native tile/core examples above are source arithmetic, not device correctness evidence.

Initial bounded B1 quality command, parent/device operator only:

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  bash "$D/run_job.sh" fused_dit_norm_b1_smoke24 \
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_fused_distributed_norm \
  --layer 0 --batch 1 --queued 4 --quality-only \
  --policy "$D/fusions20_fused_l0_b32_s33.policy.json" \
  --output "$D/fused_dit_norm_b1_smoke24.json"
```

After that passes, cover B3/17/32, both layer kinds, both actual norm weights, optional captured input/post-attention fixtures, queued50, then watcher separately with tracking1/skip0. Parent must select explicit latest policy; these example fusions20 policies are frozen compatible controls, not automatically the eventual stage default. For performance rerun the identical passing geometry/policy with no `--quality-only`, `TT_METAL_TRACE_ALLOC_TRACKING=0`, `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, and use `env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER` to disable instrumentation (watcher=0 still enables watcher). A boundary win still requires whole-decoder PCC/contribution/cache, trace, and performance validation under the final default. DiT local norms alone or the prior composite family's timing cannot settle this family's performance.

## Follow-up after the source audit: repair and parent hardware checkpoint

The preceding “repair pending” section records the original source investigation, not the current native state. AutoDebug completed exit0; its full report was joined/read and the important claims checked against source. AutoFix prepared the four-file bounded repair in `debug_fused_rms_minimal/repair.patch`; parent applied, built and installed it at a serialized checkpoint. See `debug_fused_rms_minimal/AUTOFIX.md`, `repair.cpu.json`, and `fused_rms_minimal_build25.json` for exact scope, source/binary receipts and unresolved validation.

The separate minimal helper/probe are `tt/fused_minimal_norm_candidate.py` and `tests/probe_fused_minimal_norm.py`; the DiT files remain unchanged. They preserve FP32 row-major gamma, preallocate one-core four-tile stats and normalized outputs, and include the input reshard, normalized-output conversion, activation AG and canonical projection layout in the measured boundary. The same strict quality, rebinding, queued-input and six-trace allocation checks apply. Parent's first minimal B1 smoke passed96 quality rows and the allocation positive control, with minimum PCC0.9999917724 versus control and0.9999984950 versus independent FP64 RMS. Instrumented smoke timings are disabled. This initial proof is not a full decoder integration or performance verdict.

Parent's DiT B1 measurements found input-norm boundary51.7185→43.964us and norm+projection147.1415→142.257us; post-norm51.303→44.1945us and norm+projection155.893→152.0055us. These isolated boundaries include activation AG and layout conversion. Broader geometry and whole-decoder evidence determine whether a family is retained.

A new separate `tests/probe_fused_minimal_norm_validation.py` checks exact native admission errors then valid B1/B32 fresh bindings. Source/CPU preparation passes; parent owns device execution. A patch-only optional storage-row adapter in `debug_fused_rms_minimal/storage_row.patch` addresses production persistent-L1 distribution. Native source inspection supports nonzero Y for8×1 row shards; all10 rows ×32 logical batches pass CPU geometry checks. Row1/9 B1/B32 watcher/trace evidence is still required before relying on those offsets. The frozen helper is not edited while parent hardware commands are active.
