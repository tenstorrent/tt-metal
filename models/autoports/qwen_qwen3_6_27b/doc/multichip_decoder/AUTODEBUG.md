# AutoDebug Report: TP4 full-attention decode cache update

## Observation

Command:

```text
timeout 300 python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/multichip_full_attention_smoke.py
```

Failure: `TT_FATAL Expect input_tensor to be sharded` from
`ttnn.experimental.paged_update_cache`, after
`nlp_create_qkv_heads_decode` in the multichip full-attention decode path.
This was a source-only investigation; no TT device was opened and no
implementation file was modified.

## Headline finding

The multichip loader and decode path replace the optimized baseline's required
batch-height-sharded decode-attention memory configuration with interleaved
DRAM.

- `functional_decoder.py::from_state_dict` derives a height-sharded memory
  configuration with logical shard shape `(32, head_dim)` and a grid containing
  one core per batch row, then passes it as
  `decode_attention_memory_config`.
- `optimized_decoder.py::_full_attention_decode` passes that configuration to
  `nlp_create_qkv_heads_decode`. Its Q/K normalization and RoPE helpers may
  temporarily use DRAM, but `_partial_rope_decode` explicitly converts its
  result back to `self.decode_attention_memory_config` before cache update.
- `multichip_decoder.py::from_state_dict` instead passes
  `decode_attention_memory_config=ttnn.DRAM_MEMORY_CONFIG`, and
  `_full_attention_decode` additionally hard-codes
  `memory_config=ttnn.DRAM_MEMORY_CONFIG` on `nlp_create_qkv_heads_decode`.
  Thus K and V are interleaved at the cache-update boundary (V directly; K is
  converted back by RoPE only to the same DRAM configuration).
- The paged-update device validation unconditionally requires the update tensor
  to be sharded, rejects width sharding, requires shard width equal to padded
  head dimension, row-major orientation, and requires the shard grid to have
  exactly one core per user/batch row. The observed fatal is therefore the
  exact predicted first validation failure.

Verdict: **verified source-level root cause**.

## Focused fix

Restore the baseline's workload-derived batch-height-sharded
`decode_attention_memory_config` for TP4, computed from the mesh device grid and
batch exactly as in `FunctionalDecoder.from_state_dict`, and use
`self.decode_attention_memory_config` for `nlp_create_qkv_heads_decode` rather
than hard-coded DRAM. This remains a per-device layout: for the current batch=1
smoke, each device has one local shard/user containing its local K/V head with
full padded head width 256. It does not undo mesh TP or replicate KV heads.

Do not use width sharding, and do not choose a grid with more cores than the
batch count: both violate explicit `paged_update_cache` validation. Merely
converting K immediately before the update is insufficient because V has the
same contract and subsequent decode attention/head concatenation is designed
around the shared decode-attention layout.

## Focused verification experiment

1. Add a cheap pre-update assertion/probe (temporary if desired) for both K and
   V on every device: `is_sharded`, HEIGHT_SHARDED, ROW_MAJOR, shard width 256,
   and shard-grid core count equal to padded batch/users (1 in the smoke).
2. Rerun the original command unchanged. Prediction: it passes the prior
   `paged_update_cache` validation and reaches/finishes paged SDPA. The existing
   test then verifies TP4 output PCC >= 0.995 against the serialized optimized
   baseline, identical replicas, and local cache shapes `(1, 1, 64, 256)`.
3. Run the same focused decode with batch greater than one (the stage's target
   batch) to prove one-core-per-user grid construction rather than a batch=1
   accident; inspect K/V shard specs before the update.
4. After correctness, rerun the relevant warmed trace-replay and watcher-only
   checks because the fix changes L1 allocation/sharding at a trace-critical
   cache boundary.

## Remaining uncertainty

Source inspection proves why the reported fatal occurs and identifies the
minimal contract-preserving correction. Hardware execution is still required
to detect a later, independent SDPA/concat layout issue or a Blackhole L1
capacity/program constraint. Such a later failure would not refute this cache
update diagnosis.
