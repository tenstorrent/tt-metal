# Clean metadata-tensor interface prototype

This note records the proposed non-breaking API for trace-safe scalar inputs in
`high_bw_all_gather` and `ring_indexer_score_dsa`.

The existing scalar arguments remain part of each API. The metadata path adds
parallel tensor arguments carrying exactly the same values. They are
one-element, row-major, interleaved `UINT32` device tensors. For each value, a
call selects either the scalar form or its tensor duplicate; supplying both is
invalid.

## `high_bw_all_gather`

```python
ttnn.experimental.high_bw_all_gather(
    input_tensor,
    dim,
    output_tensor,
    cluster_axis,
    *,
    input_batch_index=None,
    gathered_dim_size=None,
    input_batch_index_tensor=None,
    gathered_dim_size_tensor=None,
    ...,
)
```

- `input_batch_index_tensor[0]` is the final batch index, exactly like
  `input_batch_index`. The op does not apply user/layer linearization.
- `gathered_dim_size_tensor[0]` is the final gathered-dimension extent, exactly
  like `gathered_dim_size`. The op does not add or round a chunk prefix.

## `ring_indexer_score_dsa`

```python
ttnn.experimental.ring_indexer_score_dsa(
    q,
    k,
    weights,
    k_local,
    ag_multi_device_global_semaphore,
    *,
    cache_batch_idx=None,
    kv_len=None,
    cache_batch_idx_tensor=None,
    kv_len_tensor=None,
    ...,
)
```

- `cache_batch_idx_tensor[0]` is the final cache batch index, exactly like
  `cache_batch_idx`. The op does not apply user/layer linearization.
- `kv_len_tensor[0]` is the final valid KV length, exactly like `kv_len`. The op
  does not derive it from a chunk start.

## Compatibility

The existing scalar forms remain unchanged; the two tensor arguments duplicate
them for trace-safe replay. During migration, the current
derived-metadata implementation can remain as an internal/legacy entry point;
it must not define the semantics of the clean overloads. Once callers produce
the final slot and length tensors directly, the legacy entry point and its
layer/slab parameters can be removed.
