## Program cache review — data_movement/concat

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra with multiple sharded/interleaved variants, each returning a program and override callback.
- Hashing: default determinants cover layout, dim, grouping, sharding specs, and shapes; no runtime buffer addresses are hashed.
- Overrides consistently update:
  - For sharded tiled RM/tiled variants: dynamic CB base addresses for each input CB and the output CB.
    - References: `s2s_tiled_concat_two_tensors_height_multi_core` and `s2s_rm_concat_two_tensors_height_multi_core` override callbacks.
  - For interleaved variants: reader per-core source addresses block and writer destination base address.
    - Reference: `concat_multi_core` override updates `reader` src addrs and `writer` dst addr.
