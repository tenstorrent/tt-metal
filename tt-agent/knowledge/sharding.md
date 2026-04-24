# Sharding

Topic knowledge for tt-metal sharding — external references only today;
extend with patterns and traps as they surface.

## External references

- **Tensor sharding tech report** — `tt-metal/tech_reports/tensor_sharding/`
  Interleaved vs height / width / block sharding, when to use each, L1
  constraints.
- **ND sharding example** — search `tt-metal/ttnn/cpp/ttnn/operations/`
  for ops implementing `height_sharded` or `block_sharded` to see shard
  spec setup in practice.
- **Sharding in ttnn tests** — `tt-metal/tests/ttnn/unit_tests/operations/`
  Tests show full sharding setup including `MemoryConfig` and
  `ShardSpec`.
