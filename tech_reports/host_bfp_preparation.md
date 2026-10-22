# CPU BFP weight preparation

The host converter can read unpadded FP32 or BF16 matrix rows and write BFP4_B
or BFP8_B bytes directly into the final TTNN host buffer. This removes the
temporary tiled matrix and its copy. Other layouts keep the existing factory.

| Part | Behavior |
|---|---|
| `bfp_simd.hpp` | One row encoder, with runtime scalar, AVX2 and AVX-512 selection. The integer rounding rules match the original converter. |
| `bfp_tile_encode.hpp` | One tile traversal for consecutive tiles and pitched matrix rows. Exponent padding comes from HAL L1 alignment. |
| `bfp_host_encode.hpp` | Check shape, row pitch and source extent. Check the packed byte count against `TensorSpec`. |
| Final host allocation | Allocate without filling the buffer first. Each worker writes all encoded bytes and clears exponent padding. `MemoryPin` retains the array until the last host-buffer owner releases it. |
| `bfp_tasks.hpp` | Drain every submitted task before propagating an error. Both the SIMD and original packing paths use it. Nested calls on the same executor run inline. |
| Mesh conversion | Pack eligible column slices directly using their row pitch. Copy other views in blocks, with the existing iterator fallback. |

The vector-returning `pack_as_bfp_tiles` API still value-initializes its
`std::vector<uint32_t>`. Removing that initialization requires a different
owning-buffer interface or allocator type. The direct TTNN factory avoids it
without changing that API. No process-wide thread, affinity or NUMA policy is set.

## Correctness and CI

Row tests compare every available ISA with the original integer converter.
Factory tests use the original packing loop for their expected bytes, not the
new SIMD implementation. The tests cover padding, special values, strides,
buffer lifetime, concurrent calls, nested calls and task failures.

Unavailable ISAs report `SKIPPED`. A separate Loudbox PR/merge gate sets
`TT_METAL_TEST_REQUIRE_AVX512=1` and runs the AVX-512 case. It fails if the CPU
does not support that ISA. The general CPU job continues to test other hosts.

Blaze can compile its compressed-tensor extension against the internal row
header without linking Metal. This is a source integration, not a stable public
API. Its independent NumPy quantizer tests must pass after each source uplift.

## Related changes and merge order

These open PRs touch the same conversion path. They need an explicit integration
review; a clean Git merge alone does not establish correct behavior.

| PR | Required integration |
|---|---|
| [#44421](https://github.com/tenstorrent/tt-metal/pull/44421) | Contains an alternative BFP8 SIMD and thread implementation. Merge this shared encoder first, then rework #44421 on top. Retain one row encoder and one task scheduler. Port any remaining improvement with byte-parity and performance tests. Do not resolve the function conflict by keeping both implementations. |
| [#57298](https://github.com/tenstorrent/tt-metal/pull/57298) | Adds `optimize_bfp` exponent search. When combining the changes, add `&& !optimize_bfp` to the SIMD fast-branch guard in `pack_as_bfp_tiles`. Search requests must reach the original loop until the row encoder supports search. Test FP32 and BF16 inputs, BFP4_B and BFP8_B, with both option values. Include an input for which Emax-1 wins; a plain round-trip test is insufficient. |

The current base has no `optimize_bfp` argument. This PR does not add that feature
or a dummy flag. In #57298, Python conversion first creates a float/BF16 host
tensor when search is requested, then calls `to_dtype(..., true)`. Keep that
routing so the direct matrix factory cannot bypass exponent search either.
The merge order above is a proposal for the owners of those PRs; their approval
and validation of the combined branch remain merge requirements.

## Performance scope

The [Blaze report](https://github.com/tenstorrent/tt-blaze/blob/kpaigwar/compressed-pack-throughput/docs/compressed_tensor_packing.md)
contains complete-layer and stage timings for the two optimization PRs together.
Those measurements exclude checkpoint and cache I/O, device upload, compilation
and inference. They are not isolated Metal-only gains or full-model startup times.
