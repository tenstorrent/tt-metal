Binary OP program cache review

Summary

- Reviewed `ttnn/cpp/ttnn/operations/eltwise/binary` program cache usage across elementwise and broadcast variants (including sharded paths).
- No program-cache correctness issues were found. Overrides update buffer addresses and runtime-only values; hashing keys compile-time determinants and factory selection.

Key observations

- Custom program hash
  - Implemented in `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` under `BinaryDeviceOperation::compute_program_hash(...)`.
  - Hash includes: operation attributes (which exclude the per-call scalar), selected program-factory index, and input dtypes and memory configs. Shapes are intentionally excluded because factories configure compile-time resources independently of total size for interleaved tensors and rely on runtime args to cover sizes/offsets.

- Override runtime arguments
  - Elementwise (interleaved) variants call the common helper `set_eltwise_binary_runtime_args<false>(...)` during cache-hit, which:
    - Updates input/output buffer base addresses.
    - Computes per-core start tile indices, counts, and other runtime scalars and writes them in the exact order used at creation.
  - Broadcast and sharded specializations provide explicit overrides that update relevant circular buffer dynamic addresses (e.g., for sharded H-broadcast paths) and per-core arguments.
  - Verified representative override implementations:
    - Elementwise: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_program_factory.cpp`
    - SFPU elementwise: `.../element_wise_multi_core_sfpu_pgm_factory.cpp`
    - Broadcast H/W/HW and sharded H variants under `.../device/`

- Argument ordering and kernel alignment
  - Reader/writer/compute kernels have consistent runtime-argument vector sizes between create(...) and override(...). The common helper centralizes population to minimize drift.

Cross-check with existing tests

- Core functional and sharding broadcast behavior is heavily exercised by `tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py`.
- That suite includes a decoder-style cache population check which repeatedly reuses programs over different tensor allocations without issues.

Conclusion

- The Binary OP’s program cache usage is correct and efficient given current factory designs. No failing cache-hit tests were identified or required. This OP is marked reviewed.
