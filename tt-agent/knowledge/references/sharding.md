# Sharding Reference Pointers

## Tensor sharding tech report
`tech_reports/tensor_sharding/`
Covers: interleaved vs height/width/block sharding, when to use each, L1 constraints.

## ND sharding example
Search: `ttnn/cpp/ttnn/operations/` for ops that implement `height_sharded` or `block_sharded`
to see how shard spec is set up in practice.

## Sharding in ttnn tests
`tests/ttnn/unit_tests/operations/`
Tests often show the full sharding setup including MemoryConfig and ShardSpec.
