# BF4 KV reset: source-only AutoFix assessment

2026-10-25. No devices accessed, active candidate interrupted, or implementation edited.

**Leading minimal repair hypothesis:** in `Generator.reset`, specialize only BF4 tiled device cache fields to `ttnn.fill(value, 0.0, output_tensor=value)`; retain existing `copy(zeros_like(value), value)` for other dtypes. This is supported by the executed source chain below, but BF4 fill is absent from the binding's advertised dtype list and the inspected fill tests. Hardware validation is required before claiming it works.

## Why reset currently constructs host tensors

`generator.py:366` resets every state tensor with `copy(zeros_like(value), value)`. `creation.cpp::full_like_impl` dispatches tiled, same-dtype device tensors to `ttnn.fill` only for BF8/BF16/F32. BF4 instead reaches `full_impl`, which constructs a `std::vector<float>` of the full logical volume, packs it through `Tensor::from_vector`, and uploads it. This occurs again for each reset rather than only during initial allocation. At full context, the production per-rank cache field has `[8192,1,32,256]`: 67,108,864 float elements (256 MiB host payload before packing) per K or V field. Repeated preparation resets therefore perform substantial host work. This source finding explains a plausible cost mechanism; no timings were measured here.

## Direct fill is the strongest existing device candidate

- `eltwise/unary/unary.cpp` registers `fill` as scalar unary `FILL`. `unary_impl` accepts a preallocated output when input/output dtypes are identical, including BF4. Its BF16/F32/BF8 whitelist applies only to **mixed** dtype outputs; supplying a BF16 input with BF4 output is rejected and is not an alternative.
- `unary_device_operation.cpp` validates device storage, layout and output shape without a BF4 exclusion for FILL. It returns the supplied output tensor, avoiding cache replacement/allocation.
- `unary_program_factory.cpp` derives input/output data formats and tile byte sizes from their actual dtypes. The generic `eltwise_sfpu_metal2.cpp` reads a tile, copies it to destination registers, executes the generated fill, then packs to the configured output format. `common/unary_op_utils.cpp` generates `fill_tile_bitcast` for floating formats with literal zero bits for0.0; this overwrites values rather than multiplying potentially nonfinite inputs by zero.
- The interleaved reader/writer process corresponding disjoint tile ranges. A tile is read before its resulting output is written, so same-buffer use has no cross-tile dependency in this path. The existing model cache is tiled DRAM interleaved, not a new sharded-unary case. BF4's packed exponent/data representation is handled by the packer, not by incorrectly writing BF16 zero pages into a BF4 allocation.

Thus there is no source-visible validation blocker to same-dtype BF4 in-place fill. The documentation omission prevents calling this an already-tested BF4 API guarantee. A separate-output `ttnn.fill(value,0.0)` followed by the existing copy is a fallback discriminating control if in-place behavior fails; it still avoids the host full construction but allocates an extra cache-sized tensor.

## Rejected alternatives and trace implications

Device `moreh_full_like` calls `prim::full`. `full/device/full_device_operation.cpp::validate_inputs` explicitly permits only BF16, INT32 and FLOAT32, so BF4 device full/full-like is blocked. Public creation `full`/`zeros` uses the same host-packed BF4 branch already implicated. Merely supplying `optional_output_tensor` to `zeros_like` does not bypass that host branch. A new raw-memory zero kernel or global creation-op change is broader than the available model-local fill experiment.

In-place fill retains the existing cache address, dtype, shape and mesh distribution needed by captured decoder traces. Reset currently executes outside the model/sampler capture bodies; it already includes host input refresh operations, so the proposal does not make the whole reset capturable. Warm the fill signature during normal preparation before capture, preserve serialized CQ ordering, and verify repeated reset followed by existing trace replay. Do not rebind cache fields to a new tensor. Device-fill trace compatibility is not itself proof that aliasing, cache-hit patching or all-rank zeroing works.

## Smallest required proof

Under the parent's hardware ownership, create a small production-shaped BF4 cache with nonzero values on all four ranks; fill in place, synchronize/read back, and require every logical value to be exactly zero with unchanged buffer addresses. Repeat after different nonzero content to exercise program-cache hits. If this passes, use a real full-attention layer or the existing four-layer/full-head generator with BF4 KV: reset after a populated cache, replay existing traces, and compare repeated outputs while recording unchanged trace IDs/capture counts. Include one full-sized cache reset timing to distinguish removal of host packing from merely shifting the cost. Preserve the initial full candidate result and leave baseline BF8 reset untouched. Initial BF4 cache allocation and external-owner scratch creation may still use host zeros; this narrow repair targets repeated reset, not every allocation site.
