> Current stage outcome: [layer34 AutoFix failed to meet the unchanged gate](endurance/AUTOFIX_v5_layer34.md). The material below records earlier repair history.

# AutoFix: DRAM-sharded fused bias

Diagnosis: [AUTODEBUG.md](AUTODEBUG.md), independently produced by the installed inspection-only launcher. Its reader/caller contract was verified in the current source and by two device probes.

Failing contract: layer0 decode PCC≥0.995 with unchanged BF16 weights, HiFi4/FP32 accumulation and packed-column semantics. Original candidate gave0.967118 eager and traced PCC; prefill was unaffected. A separate K-block8 L1 allocation failure preceded this and was resolved by legal block2.

Hypotheses, in priority order:

1. Wrong bias address mapping: original interleaved bias passed into a bank-local reader. Separate-add control almost exactly recovers; matching DRAM bias shards recover bit-equal packed projections. Supported.
2. Weight/activation sharding or FP32 accumulation: the correctly sharded-bias control keeps these unchanged and is exact. Refuted for the observed probe.
3. Trace state corruption: eager and traced errors are identical, and the projection probe reproduces before recurrence. Refuted for this failure.

Minimal model repair: prepare a separate FP32 bias with the packed weight's DRAM bank grid and column shard width, padded shard height32; use only in the explicit DRAM-sharded decode projection. Preserve interleaved prefill bias. The change is confined to `tt/fused_decoder.py`; `tests/projection_probe.py` retains the discriminator. No native/shared-source edits.

Validation:

- `projection_probe.json`: bad fused-bias QKV PCC0.096165; max error15.56. Bias-free matmul plus ordinary add givesPCC>0.99999997.
- `projection_probe_sharded_bias.json`: all four packed segments and the output projection are bit-equal to the interleaved control.
- `dram_reader1_block2_fixed_0.json`: full decoder prefillPCC0.999925, eager/traced decodePCC0.9998867224, deterministic; direct functional equivalence passes0.995. Same bounded `run_decoder --layer0 --length256 --context512 --compare-functional` harness and serialchip0 regime as the failing run.
- This repaired one-reader configuration is slower (1.189735ms) than the earlier passing0.927520ms path. Correctness repair is verified; performance selection is not claimed. Other legal reader/block configurations remain work.

Native API limitation: the DRAM-sharded matmul validator admits an interleaved bias that its bank-local reader cannot interpret. This stage prepares the compatible layout explicitly. A generic native validation/rejection regression belongs to a separate shared-source change and is not implemented under this goal's file scope.

Separate new stress observation: the128-in/128-out sequence test exposes a position237 PCC failure in both the original functional decoder and the fused candidate. That is not explained away by this bias repair. `stress_dram_fixed_0.json` retains every row and shows bit-equal repeats; `routing_probe` is localizing the first divergence. Closure remains pending.


## Resolution updates

The bias repair was correct but the selected projection uses a faster bias-free multicore pipeline plus a separate add. Reader1/2/3 and pipeline3/6 comparisons are recorded in the work log; the final selection no longer needs the extra sharded bias allocation.

The independent sequential-state failure is resolved by model-local FP32 SFPU recurrence, later fused with native addcmul and binary input activations. `recurrent_sfpu_decay.json`, `stress_sfpu_0.json`, `stress_addcmul_0.json`, and `state_probe_repaired.json` record isolated and model-level evidence. The accepted functional control's native FDR is preserved unchanged as the failing control. Final combined-source validation is in progress; these repairs do not waive the remaining stage gates.

## Extended decode: reference cache precision boundary

The 1024-token extension exposed three deterministic output/contribution failures under the previously 128-token-clean FP32 recurrence. Independent source/CPU investigations found the installed `transformers.cache_utils.LinearAttentionLayer` sets cache dtype from BF16 convolution state (lines926/937) and `copy_` rounds each recurrent-state update (line989), despite FP32 recurrent computation. The TT buffer retains FP32 across calls. CPU replay with identical inputs and different cache precision reproduces 2.7–2.9% state error and ~0.92 error-direction cosine alignment to the observed discrepancy. Exact HF output processing of saved TT core reproduces failing attention PCC0.98781; gatednorm rounding alone gives0.999995. See both hypothesis reports and `state_probe_1024` artifacts.

Discriminating repair: round consumed state BF16→FP32 at a public decode boundary while keeping existing FP32 state allocation and internal prefill computation. `stress_cache_boundary_1024_0.json` passes all3072 outputs, minimum output PCC0.998485278/contribution0.998021592, three bit-identical trace repeats. Context capacity is unchanged; this is reference-cache semantic matching, not a capacity-reducing precision experiment. Continuation-prefill boundaries also require the same rounding exactly once per public call, without introducing extra rounding between internal chunks.

The BF16 L2, A/B/beta projection boundaries, FP32-summed convolution followed by BF16 preactivation rounding, and all three combined did not independently resolve the failure. Their exact source snapshots and failed JSONs are retained. `recurrent_fused_decay_1024.json` confirms the selected fused-SFPU EXP/addcmul recurrence itself has negligible analytic decay error over1024 steps. Native FDR with the corrected BF16 entry boundary still fails (`stress_native_cache_1024_0.json`, minimum output PCC0.961638); intermediate TF32 narrowing remains a separate defect. No native sources, HF reference, checkpoint, or gate threshold were modified.
