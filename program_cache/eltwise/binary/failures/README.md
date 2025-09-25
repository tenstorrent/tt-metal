Eltwise Binary Program Cache Failures

- test_binary_cachehit_missing_shape_in_hash.py: Expected to fail on cache-hit path due to shapes omitted from the program hash while reader/writer TensorAccessorArgs are compiled at creation.

Case: Shapes not in program hash; stale compiled TensorAccessorArgs

- Hash omits shapes:
```286:317:ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp
return operation::hash_operation<BinaryDeviceOperation>(
    attributes,
    program_factory.index(),
    input_tensor_a.dtype(),
    input_tensor_a.memory_config(),
    input_tensor_b->dtype(),
    input_tensor_b->memory_config());
```

- Program creation encodes accessor args, not shape-agnostic:
```139:158:ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_program_factory.cpp
std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
...
KernelHandle binary_reader_kernel_id = CreateKernel(... ReaderDataMovementConfig(reader_compile_time_args, ...));
KernelHandle unary_writer_kernel_id = CreateKernel(... WriterDataMovementConfig(writer_compile_time_args, ...));
```

Repro:
```bash
pytest -q program_cache/eltwise/binary/failures/test_binary_cachehit_missing_shape_in_hash.py::test_eltwise_binary_program_cache_missing_shape_in_hash -s --disable-warnings
```

Suggested fixes:
- Include shapes (or at least Ht and Wt) and device id in compute_program_hash(...).
- Or make reader/writer shape-agnostic by moving any shape-dependent addressing into runtime args.
