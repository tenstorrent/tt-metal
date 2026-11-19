# Optional transient ordered-weight L1 staging

Source-only assessment; no F edit or device execution. **Legal interface, uncertain performance benefit.** This differs from the now parent-validated product placement: staging weights still requires the same29,360,128-byte DRAM read per call and adds an L1 write/read plus a kernel dispatch. It is not another DRAM-byte elimination.

Inside the existing singleton ordered-projection branch only, after choosing the BF16 `weight`, the candidate is:

```python
weight = ttnn.to_memory_config(weight, ttnn.L1_MEMORY_CONFIG)
products = ttnn.multiply(weight, coefficients, dtype=ttnn.float32,
                        fast_and_approximate_mode=False,
                        memory_config=ttnn.L1_MEMORY_CONFIG)
del weight  # Local staging handle; setup's persistent weight stays in DRAM.
result = ttnn.experimental.fast_reduce_nc(products, dims=[0, 1], compute_kernel_config=self.compute)
```

`core/to_memory_config/to_memory_config_op.cpp` routes interleaved DRAM→interleaved L1 to native `prim::copy`, preserving input dtype when none is requested. `data_movement/copy/device/copy_device_operation.cpp` accepts BF16 and selects `DefaultTilized` for the changed memory config. `copy_default_tilized_program_factory.cpp:82–115,209–217` performs a direct reader→writer transfer with no compute conversion for equal formats; its input double buffer is two BF16 pages (4KiB/core). Thus checkpoint weight bits, shape `[4,512,32,224]` or `[2,1024,32,224]`, tile layout, and contraction order are preserved. Binary multiply's interleaved validation accepts either buffer type independently for A/B/output, and its generalized reader accesses the actual L1 A buffer directly. No reshard or host staging is required.

At110 L1 banks, weight and product each have14,336 pages. Weight reserves131×2048=268,288 bytes (262KiB) per bank; FP32 product reserves536,576 bytes (524KiB). Concurrent tensor storage is804,864 bytes (786KiB) per bank. Add the scalar-broadcast multiply's approximately20KiB CB footprint:806KiB on its active cores. If the weight remains live into first reduction, add74KiB reduction CBs to786KiB=860KiB. Releasing its local handle after multiply shortens this lifetime; underlying command dependencies still follow normal TTNN tensor lifetime rules. These exclude live model tensors, program/static reservations, and allocator fragmentation; device allocation/tracing remains the capacity discriminator.

Potential benefit is scheduling, not lower traffic: a standalone copy might stream weights more efficiently while the subsequent multiply reads only L1. Source does not prove that advantage. The copy uses its generic interleaved page reader and double buffering, not a promised bank-optimal bulk-copy engine; binary's scalar-broadcast reader already processes the weight tile stream. An extra stage can therefore lose despite capacity feasibility. Compare the **complete staging+multiply+reduce trace** against the validated product-L1 control, preserving the current output/reduction placement. Never compare only multiply after excluding its staging cost.

Keep multirow prefill DRAM, including its128-row product. Do not persist the staged weight in model state or charge it as zero-DRAM reuse: that would be a different lifetime/capacity candidate and would affect all later kernels. If temporary staging is slower, the source offers no reason to retain it merely because L1 accesses look faster individually.
