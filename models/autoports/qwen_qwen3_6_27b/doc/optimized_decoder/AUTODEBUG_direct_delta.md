# Direct S=1 recurrent GDN candidates

Source-only, 2026-11-12. `candidates/direct_delta_decode.py` is syntax checked, not device tested; main source unchanged.

## Why this is a discriminating candidate

Native chunk scan explicitly computes `v_new = Tinv @ (v_beta + nkd @ old_state)` and `out = q_decay @ old_state + intra @ v_new` (`chunk_gdn_scan.cpp:9-11`). The direct recurrent kernel instead decays state, predicts `k @ decayed_state`, forms `beta * (v-prediction)`, updates the state, and reads `q @ updated_state` (`fused_recurrent_gated_delta_rule.cpp` compute kernel, lines192-234). All CBs are FP32 and the updated-state rank-one product accumulates directly onto decayed state in DST. This removes the chunk output's two-term cancellation path while preserving algebra. It is a useful hypothesis given the stage owner's reported sequential PCC gap; inspection does not establish that it closes the gap. The direct native kernel still uses matmul reductions and can differ from the functional elementwise sum reduction.

## Legal native waves

`FusedRecurrentGatedDeltaRuleProgramFactory` asserts BH<=grid.x*grid.y (line67), but has no larger-batch waves internally. On110 cores, process two users=96 value heads per call; B32 takes16 calls. Odd B uses one user/48 heads in the last wave. Each call uses public inputs q/k `[waveB,1,16,128]`, v `[waveB,1,48,128]`, g/beta `[waveB,1,48]`, state `[waveB,48,128,128]`.

Normalize Q/K in FP32 with the functional epsilon1e-6 **before** calling the public op. `use_qk_l2norm=False` is mandatory: the public op does not implement normalization. Its documentation's “on host” means normalization is outside the fused kernel; it need not leave the device. The prototype uses device `multiply/sum/add/rsqrt`, then the public wrapper performs device head split, GQA expansion, query scaling and exp(g). No torch/from_torch/to_torch appears in decode. All native tensors/CBs are FP32; pass HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False as the precision control.

Each96-head state slab is6 MiB, with16 FP32 tiles/head. Factory circular buffers allocate roughly520 KiB/core at K=V=128 and T=1, below Blackhole physical L1; final allocator headroom still needs validation. Native public preprocessing also moves padded head tensors and has no supplied output tensor argument, so rank this candidate by full traced decode, not kernel count alone.

## Ring mode cannot alias a wave into the full B32 state

Ring mode returns its initial_state allocation directly and is genuinely safe for a local ring: reader reads the entire head's initial state and waits for a NoC barrier before publishing it; writer waits for produced outputs/state. Different heads own disjoint blocks. For T=1, `idx[h]` must equal h, because watcher asserts `idx[h] % BH == h && idx[h] < T*BH` (`reader_fused_recurrent_gated_delta_rule.cpp:71-86`). Device validation requires exact state shape `[T*BH,K,V]`, not a larger backing tensor. Therefore a96-head wave cannot pass the1536-head full state plus offset indices; neither a head offset nor a caller-supplied final-state tensor is exposed.

A regular `state[start:end]` is **not an alias view**: slice allocates an output tensor (`slice_device_operation.cpp:413-423`). `ttnn.view` preserves full logical volume and exposes no offset (`view.cpp:28-41`). Ring mode on that slice updates the slice copy only; it does not update caller state. It could save one temporary6 MiB native output allocation with setup-time arange96/arange48 index tensors, but still requires copying/scattering the slab back. The initial prototype uses ordinary final-state mode to avoid conflating two experiments.

## Preserve stable state without a whole-state concat/copy

A source-supported scatter exists: stage each final FP32 TILE slab into HEIGHT_SHARDED L1, one head/core, shard `[128,128]`; call `ttnn.experimental.slice_write(staged, state, [start,0,0,0], [end,48,128,128], [1,1,1,1])`.

- The tiled-sharded factory accepts matching input/output dtypes, including FP32 (it derives tile bytes from dtype), TILE input/output, rank4 input, shard height tile aligned, final-dim start0, and second-last start tile aligned. The prototype satisfies each condition.
- This path returns the original output tensor (`slice_write_device_operation.cpp:120-123`) and writes only the selected user slab; source/target allocations are disjoint. Preserve original `state` identity across every trace replay.
- **Do not call slice_write directly with interleaved TILE updated state.** Its public wrapper sets `rm_only_not_sharded`, converts the entire output state to ROW_MAJOR, then replaces it with a re-tilized allocation (`slice_write.cpp:34-40,126-134`). That both moves the full state per wave and violates stable allocation ownership. The deliberate L1 staging selects the safe tiled-sharded branch.

The helper's `scatter_state=True` takes this route. `False` collects slab outputs, concatenates once, and copies once into original state as an easier correctness control. Neither route is a measured improvement yet.

## Lower-risk alternative: exact functional order with final add writing state

The helper also provides `direct_vectorized`, matching functional `_delta_step`'s FP32 operation order:

`decayed = state * exp(g)`; `prediction = sum(decayed*k,K)`; `delta = (v-prediction)*beta`; `update=k*delta`; `add(decayed, update, output_tensor=state)`; `out=sum(state*(q*scale),K)`.

This is independent of head/core counts, so B32 is one batched graph. The `output_tensor` final add removes the separate96 MiB state copy and final-state allocation while preserving the reported functional arithmetic path. Both add inputs are separate temporaries; the destination state is no longer read by this add, so no input/output overlap hazard exists. Output reads occur later on the ordered queue. The final state address/object remains unchanged.

It still streams large FP32 state-sized intermediates through DRAM (at least decayed state, prediction product, outer update, output product), and is not automatically fast. Yet this is the first candidate to measure when functional recurrence passes: it makes the smallest numerical change and deletes the measured full-state copy. Further fusion should be tested separately after the sequential gate passes.

## Verification sequence

1. Run vectorized direct with final-add output_tensor and confirm same real-weight sequential per-user PCC as functional `_delta_step`, all state values, repeated trace and allocation identity.
2. Run native two-user waves with ordinary final-state mode and concat/copy control. Compare state and output per step to direct vectorized, localizing first divergence; do not assume mathematically equivalent means acceptable PCC.
3. If native passes, compare tiled-sharded slice_write scatter with concat/copy, including odd batch, changed user inputs, state resets and traces. Require bit equality of scatter versus control outputs/state, disjoint user ownership, allocation tracking and watcher separately.
4. Re-measure full traced decoder under the chosen numerical policy and the same baseline regime. The user goal still requires final performance to beat the best correct path; lower accuracy cannot be traded for speed.
