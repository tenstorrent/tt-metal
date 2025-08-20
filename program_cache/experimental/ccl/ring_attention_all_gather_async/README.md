# Program cache review — experimental/ccl/ring_attention_all_gather_async

Status: Reviewed — no blocking program-cache issues identified.

## Summary
- Uses old infra with `override_runtime_arguments_callback`.
- Overrides update:
  - Reader: semaphore addr at index 9; per-input tensor/input-output buffer addrs starting at computed offset.
  - Writer: semaphore addr at index 11; per-output buffer addrs starting at computed offset; fabric connection args depend on presence of neighbors.
- Hash includes dim, num_links, ring_size, output_mem_config, topology, cluster_axis, sub_device_id, and input tensor props — adequate to select program variants.

## References
- Program and override: `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_program.cpp`.
- Hash: `.../ring_attention_all_gather_async_op.cpp::compute_program_hash`.

## Notes
- Device routing is embedded in kernel args; hash relies on `ring_size`, `cluster_axis`, and input props. Consider hashing a stable device ordering if submesh composition can change across runs with same shapes.
