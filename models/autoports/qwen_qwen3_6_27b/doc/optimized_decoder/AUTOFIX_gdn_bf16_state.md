# BF16 persistent state with FP32 readout

Source-only candidate: `candidates/gdn_sfpu_bf16_state.py`, based on retained-no-copy. State dtype selects the path: FP32 preserves the previous schedule; BF16 enables persistent BF16 state while retaining FP32 arithmetic and output. No device execution or C++ compilation was performed by this helper.

The stage owner reports that external state rounding reduced failures in a512-step test from11 to2; adding unsharded normalization left one case at PCC0.992493. This motivates matching the cache boundary efficiently, but does not establish that BF16 storage fixes all accuracy failures.

## Exact boundary

For each value-column, BF16 input state is widened into FP32 DST0..3. Decay, prediction, delta update, and Q readout use the existing separate FP32 operations. The output is fully computed in FP32 DST6 **before** state rounding. Only then are DST0..3 rounded to BF16 and packed to the state buffer. Thus the current output uses unrounded updated state and the following decode step consumes rounded persistent state.

Rounding explicitly invokes the same `typecast_tile_init<Float32,Float16_b>` and `typecast_tile<Float32,Float16_b>` as TTNN typecast. `hw/inc/api/compute/eltwise_unary/typecast.h:160` dispatches this pair to `calculate_typecast_fp32_to_fp16b`; Blackhole `ckernel_sfpu_typecast.h:273` performs round-to-nearest-even using0x7fff plus the retained tie bit and masks the low16 bits. Its comment explicitly explains why historical pack truncation is not equivalent. This candidate does not rely on an unverified packer rounding assumption.

## Storage and formats

Only input-state CB5 and updated-state CB11 become BF16, with2048-byte pages. The reader and writer use that tile size for state page addressing and local offsets; all other CBs, including output CB12, remain FP32 with4096-byte pages. The output tensor remains FP32. Both state accessors continue to reference the caller's same allocation.

Reader compile-time argument7 is state tile bytes; TensorAccessor groups start at8. Writer compile-time argument0 is state tile bytes; groups start at1. Compute argument1 enables BF16 state, while argument0 retains normalized/raw Q/K selection. The host determines these values from `state.dtype` and validates FP32 or BF16.

The compute descriptor retains full-DST synchronization and FP32 destination accumulation. BF16 CB5/11 use `UnpackToDestMode.Default`; every FP32 CB retains explicit direct FP32 unpack. The copy source is explicitly reconfigured to CB5 before each four-state-tile load, then back to FP32 CB3 before Q/K/V/gate loads. Both `reconfig_data_format_srca` and `copy_init` are issued, because changing copy operation setup alone does not change unpack data format.

After the FP32 readout and RNE step, packer format switches from FP32 CB12 to BF16 CB11, packs the four state tiles, then restores FP32 CB12 before packing output. State packing uses a cleared edge mask; output packing reinstates the reduced-row mask. State and output publication remain after all four value-columns complete, and the existing writer waits for both.

At B32, persistent recurrent state storage decreases from96MiB to48MiB; per-token state DRAM read/write decreases from192MiB to96MiB. Input/output state CB storage decreases by64KiB/core. The candidate replaces external full-state FP32→BF16→FP32 operations with local DST rounding and direct BF16 persistence. Latency savings are unmeasured; format changes and RNE still cost device work.

## Validation and integration

Python AST and inline-brace checks passed. Source inspection verified page-size propagation, selective BF16 CB formats, post-readout RNE, source-format restoration, and packer restoration. Device compilation and correctness remain pending. First use the isolated own-input probe with BF16 initial state, compare persisted state against CPU BF16-rounded updates, and compare output against the unrounded FP32 update. Then run repeated/512-step, trace changed-input, watcher, and latency checks against the best correct candidate.

The existing caller must allocate or initialize BF16 state before selecting this path; the helper does not convert an existing FP32 allocation or introduce host conversion. Prefill/state initialization must establish the same intended cache boundary. If selected for final delivery, document the state dtype and capacity impact in the stage context contract. No context capability should be reduced: this storage change lowers recurrent-state memory usage. The independent remaining accuracy case still requires investigation.
