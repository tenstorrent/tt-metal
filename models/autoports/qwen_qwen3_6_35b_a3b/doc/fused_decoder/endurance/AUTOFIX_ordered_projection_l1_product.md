# Ordered single-token projection product in interleaved L1

Source-only assessment, no runtime edits or device execution. **The native interfaces admit this candidate without a reshard or runtime host conversion.** The first experiment can change only the single-token product's output memory config; leave weights, coefficients, reductions, output placement, arithmetic, and multirow prefill unchanged.

```python
# Only inside linear()'s x.shape[-2] == 1 / attention_packed / linear_attention branch:
products = ttnn.multiply(
    weight, coefficients,
    dtype=ttnn.float32,
    fast_and_approximate_mode=False,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
result = ttnn.experimental.fast_reduce_nc(
    products, dims=[0, 1], compute_kernel_config=self.compute
)
```

Do not set L1 memory on the multirow product. The first branch also serves single-token internal prefill tails with `projection_k=1024`; their product has the same total size, and the unchanged dimensional reduction retains their existing public-prefill arithmetic order. There is no need to add a decode-only flag merely for this size distinction.

## Native contracts

- `binary_ng/device/binary_ng_device_operation.cpp:392–422` admits nonsharded INTERLEAVED input/output, without restricting the buffer type to DRAM. Explicit output memory chooses the new allocation; both existing operands remain DRAM. For MUL, `fast_and_approximate_mode=False` selects the accurate SFPU implementation (`:51–68`), independently of this placement change. FP32 result packing and broadcast dimensions stay unchanged.
- Binary factory `binary_ng_program_factory.cpp:1350–1360` builds `TensorAccessorArgs` from the actual output buffer, so its writer routes an interleaved L1 destination directly. No sharding adapter is required. Interleaved output still uses the ordinary worker-grid split; changing memory location does not request a different arithmetic operation.
- `fast_reduce_nc_device_operation.cpp:51–104` permits FP32 TILE input and reduction over non-spatial dimensions, and invokes common reduction validation. Neither validation requires DRAM input. The factory binds the actual input TensorSpec to `FRNC_INPUT` (`fast_reduce_nc_program_factory.cpp:185–193`); `reader_reduce_nc.cpp:43–84` constructs `TensorAccessor(tensor::src)` and reads its pages through NoC. This is the current generalized-accessor version, rather than a DRAM-only reader. It supports direct remote-L1 interleaved reads.
- Keep default reduction output memory (DRAM) for this discriminator. `fast_reduce_nc_nanobind.cpp:65` defaults `memory_config` to `DEFAULT_OUTPUT_MEMORY_CONFIG`; both outer stages retain FP32. The first intermediate is only28 tiles/114,688 bytes for projection_k512, and the final result7 tiles/28,672 bytes. Moving those too would confound the first trial and is unnecessary to remove the large product's DRAM round trip.

## Capacity estimate including buffers

Default product shape is `[4,512,32,224]`, all inner dimensions tile-aligned:14,336 FP32 tiles,58,720,256 bytes. The alternative singleton prefill path `[2,1024,32,224]` has the same size. Interleaved allocation reserves `ceil(num_pages/B)*aligned_page_size` per bank (`tt_metal/impl/allocator/allocator.cpp:785–793`), where **B is the runtime L1 bank count**, not necessarily a hardcoded core count. At110 banks this is131×4096=536,576 bytes=524 KiB per bank, slightly above the average533,820.5 bytes.

For this BF16×BF16 scalar-broadcast-to-FP32 multiply, the native factory's principal CB allocations are two BF16 pages for each operand, two BF16 broadcast pages for B, and two FP32 output pages:20 KiB/core (`binary_ng_program_factory.cpp:1182–1287`). The optional row-broadcast column cache does not apply to this scalar broadcast. Any compiler-inserted input-activation scratch would add only tile-scale storage; it is not proportional to the full product. The materialized product is an ordinary L1 allocation, **not** a whole-product CB.

The first reduction's input granularity is8 (largest divisor≤8 of512 or1024). Its CB/DFB budget is16 FP32 input pages +1 BF16 zero page +2 FP32 output pages =75,776 bytes=74 KiB/core. The obsolete accumulated-sum buffer is explicitly absent (`fast_reduce_nc_program_factory.cpp:117–133,200–226`). Therefore product+first-reduction local buffers account for598 KiB on each active reducer core at110 banks. Only28 reducers are needed for projection_k512 (14 for1024); the interleaved product remains spread over all L1 banks, so those reducers read many remote-bank pages.

This fits comfortably below the nominal1.5 MiB/core figure **in isolation**, but is not an allocation guarantee. Actual available space is `worker_l1_size - l1_unreserved_base - l1_small_size`, further reduced by live model tensors, program allocations, and fragmentation (`l1_banking_allocator.cpp:119–157`). The correct capacity check is the complete warmed/traced decoder and its actual allocator geometry, including the current L1 intermediate lifetimes. Product is local to `linear()` and released when it returns; do not retain it in model state or diagnostic hooks across later matmul stages. No allocation-size failure has been observed for this candidate by this investigator.

## Traffic and measurement discriminator

If these native kernels execute as their accessors specify, the58,720,256-byte product write and same-sized first-reduction read move from DRAM to L1/NoC:117,440,512 bytes of operation-derived DRAM traffic removed per single-token ordered projection. Useful projection weight/cache bytes do not change, so this does not increase the useful-byte roofline numerator. The minimum weight read (ordered BF16 padded weights29,360,128 bytes) still comes from DRAM. Native Q/Z and all multirow-prefill products remain unchanged.

Moving the traffic does not remove its writes/reads or guarantee proportional latency improvement. The reducer uses relatively few cores and performs remote-L1 reads; NoC placement and allocator-bank mapping may dominate. Root should compare actual same-input outputs and warmed full trace latency, then retain the existing endurance/continuation and mixed-length checks if faster. Multirow prefill products are unsuitable:128 rows would require7,516,192,768 bytes (~68.3 MB per110-bank average), far beyond L1. This proposal is restricted to the existing singleton branch.
