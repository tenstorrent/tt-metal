# Inline FP32 SFPU recurrent GDN prototype

Source-only AutoFix hypothesis helper, 2026-11-12. The stage owner reported that native recurrence loses roughly0.11% state magnitude per step, whereas direct SFPU recurrence agrees with its own-input CPU reference to approximately1e-8. Those measurements belong to the owner's evidence; this helper did not run hardware.

`candidates/gdn_sfpu_inline.py` contains a complete Python `ttnn.generic_op` builder and inline reader, compute, and writer source strings. Python syntax compilation passed. Device compilation, numerical behavior, trace replay, watcher, and performance are **unverified**.

## Math and work distribution

The program performs direct S=1 recurrence entirely through FP32 SFPU arithmetic:

1. Decay the16 state tiles using scalar SFPU multiplication.
2. Multiply state tiles by the K column vector with SFPU column broadcast; fold four H tiles by SFPU addition; reduce32 rows with `sfpu_reduce<SUM,Float32,REDUCE_COL>`.
3. Form `(v-prediction)*beta` using SFPU subtraction and multiplication.
4. Broadcast delta rows and K columns to form the outer update in DST, then SFPU-add decayed state.
5. Read updated state with the scaled Q column, using the same accurate SFPU contraction.

There is no FPU matmul, multiply, or reduction. **The first prototype nevertheless truncated FP32 during unpack; the precision correction below is required.** With that correction, copy/pack retains FP32. The accurate reduction follows the same fold-then-reduce topology as `kernel_lib/reduce_helpers_compute.inl`; the generic reduction wrapper explicitly distinguishes this accurate SFPU route from the fast TF32 FPU route (`operations/reduction/generic/device/kernels/compute/reduce.cpp:22-24`).

One program distributes BH heads across up to110 cores. Each core loops its contiguous head range, so B32's1536 heads take14 or15 heads/core with no host wave loop. State traffic is one16-tile read and one16-tile write per head. All intermediate state remains in local circular buffers. The initial candidate allocates76 FP32 tiles/core,304KiB, plus ordinary kernel/runtime overhead.

## Calling contract

All inputs must be FP32 TILE interleaved DRAM:

| Input | Shape | Meaning |
|---|---|---|
|qcol|`[B*48,128,1]`|Normalized Q, repeated for GQA and already multiplied by `128**-0.5`|
|kcol|`[B*48,128,1]`|Normalized K, repeated for GQA|
|v|`[B*48,1,128]`|Value projection after convolution/SILU|
|decay|`[B*48,1,1]`|`exp(g)`|
|beta|`[B*48,1,1]`|Beta gate|
|state|`[B*48,128,128]`|Metadata reshape of the caller's persistent state|

`recurrent_sfpu(..., output=None, cores=None)` returns FP32 `[B*48,1,128]`. The optional output tensor can be preallocated before capture. The state tensor is updated in place and its allocation is never replaced. Normalization, GQA expansion, Q scaling, head-major layout, and exp(g) remain external device operations for this first prototype. Measure their full cost.

## Ownership and synchronization

The reader loads a whole head's initial state and waits for a NoC read barrier before publishing it. Each head belongs to exactly one core. Thus the state writer cannot overwrite data before its own head has read it, and cannot affect another head.

Compute retains updated state locally while calculating readout. The writer waits for **both** updated state and output before writing/popping either circular buffer. This prevents the writer from advancing the updated-state CB read pointer while compute is still reading it. The writer uses the explicit CB `READ_PTR` selector, as the existing native recurrent writer does. All NoC writes complete before those CB pages are recycled.

## Runtime addresses and generic-op cache

Python `KernelDescriptor`/`RuntimeArgs` bindings accept uint32 runtime values and do not expose C++ `emplace_runtime_args(Buffer*)`. Therefore the prototype rebuilds a descriptor from the current tensor addresses on every call and passes all six inputs plus output through `generic_op`'s `io_tensors` list. The final entry is the preallocated output, as generic-op requires.

This uses the supported safe descriptor rebuild path: generic-op's factory is a `ProgramDescriptor` factory, and without declared buffer bindings the cache-hit branch rebuilds and calls `apply_descriptor_runtime_args` (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:819-824`). Kernel runtime **values** are excluded from generic-op's program hash, but fresh values are copied on each dispatch. Trace capture records that invocation's addresses. Do not cache the whole Python descriptor with stale addresses or claim tensor-bound Python runtime arguments exist.

Run the allocation-tracking positive control and changed-input/changed-state cache-hit test anyway; source analysis does not replace those gates.

## FP32 unpack correction

The stage owner subsequently measured a long random-sequence minimum PCC of0.96987576 for the first `sfpu10` implementation. This invalidated the assumption that FP32 CB storage plus FP32 destination accumulation preserved the input precision through `copy_tile`.

Source confirms the missing setting:

- `tt_metal/jit_build/genfiles.cpp:1069-1075` selects `unpack_conditional_dst_format=Tf32` for an all-FP32 kernel with `fp32_dest_acc_en=True`.
- `tt_metal/jit_build/data_format.cpp:211-217` overrides the individual CB's unpack destination to `Float32` only when that CB has a non-default `unpack_to_dest_mode`.
- Standard binary SFPU does set `UnpackToDestFp32` on input and intermediate CBs (`binary_ng_program_factory.cpp:1372-1400`). Accurate FP32 reduction tests do the same (`tests/ttnn/unit_tests/kernel_lib/reduce/test_reduce_helpers.py:410-423`).

Both prototype helpers now construct their config with `_fp32_compute_config()`: a64-entry mode vector, default for unused CBs, `UnpackToDestFp32` for every used FP32 CB. This is a **post-construction property assignment**, because Python's config constructor does not accept the mode-vector keyword. The original helper covers CB0..12; compact covers CB0..18, including all intermediate state and normalized-vector buffers. Python syntax compilation passed after this correction; corrected device accuracy remains for the stage owner to measure.

On Blackhole, `UnpackToDestEn` is already true in LLK definitions (`tt_llk_blackhole/llk_lib/llk_defs.h:147`); the per-CB format selects whether a32-bit input actually takes that route. `copy_tile` and its LLK datacopy path then use the direct-to-DST handshake for the full32-bit format. Raising destination precision or fidelity alone does not make this format change.

## First device checks

Start with B1,48 heads and supplied random normalized inputs against the direct SFPU own-input CPU reference. Verify state/output independently, then128 sequential steps, B32, odd batch, and distinct per-head sentinels. Compare eager and trace, persistent state buffer address, allocation tracking, and a separate watcher run. Once correctness passes, measure full-decoder traced latency with all preprocessing included. A compile/API failure should be repaired against the local APIs, rather than treated as evidence against the topology.
