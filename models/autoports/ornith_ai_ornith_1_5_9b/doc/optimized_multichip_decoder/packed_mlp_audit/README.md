# Native-residual packed MLP discriminator

Source-only; hardware compilation, PCC, watcher and latency are pending. Production and queued modules are unchanged.

The current B32 two-kind profile spends118.391 us in four separate gate/up matmuls. Earlier `multichip_optimization.py:130–155` packed candidates override single-row `finish`, but current batched decode calls `decode_finish`. Existing batched fused-gather MLP changes residual/CCL topology simultaneously. This candidate closes the standalone current-native-path comparison.

`tests/multichip_native_packed_mlp_candidate.py` subclasses current production. It changes only MLP setup and `decode_finish`, plus dispatches private one-row `finish` through that same implementation. Prefill rows>1, attention, state pools, CCL and layer-boundary residual contract remain inherited. A one-row prefill tail still returns one logical row and uses the selected MLP, so its correctness remains part of the ordinary harness gates.

| Variant | Producer and consumer contract |
|---|---|
| `packed_sharded`, readers8 | Build per-rank `[gate3072,up3072]` BF8 weights, DRAM width shards over eight banks; packed BF16 output on96 local cores with64 columns/core. Two native tiled slices write gate/up directly onto the existing48-core,64-column down-input layout. Existing BF16 SiLU/multiply; down projection unchanged. |
| `packed_sharded`, readers2 | Same complete layout family with a different legal reader/pipeline count. |
| `packed_interleaved`, readers8 | Packed output on64 cores with96 columns/core; explicit interleaved split then reshard to the existing down-input layout. This is an alternate legal adaptation/control, not a replacement for testing native slicing. |
| `minimal`, K16 | Existing alternating gate/up32-column weight tiles feed `minimal_matmul(fuse_swiglu=True)`, M1/K16/N8, same LoFi/FP32-destination compute config, BF16 L1 output then explicit conversion to the existing down-input layout. |
| `minimal`, K8 | Smaller K-block adaptation of the same fused family; retain every correctness gate. |

Packed weights have6144 columns:24 tiles/bank, divisible by1/2/4/8 readers; K4096 is128 tiles, input shards are32×128 columns and block2 is exact. Source: DRAM-sharded factory `...matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:1314–1368` validates full-K bank shards, computes column partition/pass counts and bounds work. Width96/64 output layouts each cover6144 exactly and fit110 target cores. Output storage width is independent of compute-reader width; the factory carries explicit source/output core maps (`:1411–1431`). No raw address view is used across halves.

Native slice is source-supported: `ttnn/.../data_movement/slice/device/slice_device_operation.cpp:455–456` routes TILE input through a TensorAccessor factory supporting sharded buffers. Both endpoints3072/6144 are tile-aligned; output logical height remains B, physical height32. The interleaved family remains available if a device-specific native-slice defect needs isolation. Do not reject the entire packed family on the first shape/layout error.

Packed weights retain the original BF8 block boundaries: gate/up widths are multiples of32, so concat does not mix block quantization groups. Both matmul outputs and subsequent SiLU/multiply retain BF16 boundaries. **Minimal fused SwiGLU is numerically distinct:** `minimal_matmul/device/kernels/swiglu_sfpu.hpp:46–72` chooses the FP32 sigmoid/combined expression when FP32 destination accumulation is enabled. It does not promise the separate matmul BF16 pack plus BF16 SiLU intermediate. This is a measured precision candidate, not an identity proof; unchanged .995 output/contribution/state/long-answer gates determine acceptability. Failure there is distinct from failure of packed separate-activation variants.

The experimental packed copies add26738688 DRAM bytes per layer per chip while keeping existing split weights for inherited paths. Active decode weight traffic is the same two original matrices, not doubled. No context limit changes here. Promotion must either remove unused duplicate copies safely or account for the extra all-layer storage in context_contract; minimal uses existing packed-prefill weights and adds no weight storage.

`../native_packed_mlp_jobs.json` has25 fresh serialized jobs: five families × B1/B6/B32 short state+benchmark, B32 prefix4095/1024-answer, and separate B32 watcher/state checks. It reuses existing state_stack_{1,6,32}.pt and answer1024_state_stack_32.pt references, retains immutable program-cache tracking, uses packed I/O, and leaves watcher outside benchmark jobs. Job command/source snapshots identify the actual production base. Parent should finish a candidate batch before changing that base or prepare new job labels for a rebase.

Host checks passed: AST and help before opening devices, eight host packed configs and two minimal configs,64 mocked logical split/consumer checks covering B1..32 in both packed layouts,25 unique unexecuted labels, all reference files present. These validate host scheduling/layout intent, not device math. `source_manifest.json` and `host_validation.json` retain details.

Parent launch, only through its serialized hardware protocol:

```bash
PYTHONPATH=. python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.run_stage5_jobs models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/native_packed_mlp_jobs.json
```

Audit and advice mapping: `../finalization_gap_audit.md` contains the whole-target roofline arithmetic. Exact text advice is `tracy_postloop_target/stack32/decode_perf_report.txt:84–89`, GenericOp115392,9 us preceding gap and3 us estimated opportunity. That row is ordered recurrent/postnorm, not these gate/up matmuls. Fast-zero, ordered-loop and identity work target its kernel/dataflow cost; a final profile must show whether its preceding gap also changes. The profile is already traced, so repeating the generic tracing advice is not additional work. Empty advice CSV does not mean the text contains no advice.
