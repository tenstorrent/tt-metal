# reader_mcast_transformer_group_attn_matmul.cpp — DEFERRED (design gap)

## Verdict: deferred (helper design gap — NOT migrated, file untouched)

Rotating-role per-tile-row sender/receiver (like the migrated block_sharded kernel). The migration
blocker is the **compile-time-sem-id design gap**:

```cpp
Semaphore<> sender_sem(get_arg_val<uint32_t>(i++));    // line 54
Semaphore<> receiver_sem(get_arg_val<uint32_t>(i++));  // line 55
```

Both semaphore IDs are **RUNTIME args** (`get_arg_val`), supplied per-core by the host. The v7
`SenderPipe<NOC_ID, DATA_READY_SEM_ID, ..., CONSUMER_READY_SEM_ID, ...>` and
`ReceiverPipe<DATA_READY_SEM_ID, ..., CONSUMER_READY_SEM_ID, ...>` take the sem ids as **compile-time
template params**. A runtime sem id cannot be passed as a template argument — inexpressible in v7
(known compile-time-sem-id gap). Needs helper change (out of scope).

Note: the block_sharded rotating-role sibling that WAS migrated uses compile-time sem ids
(`get_compile_time_arg_val`), which is why it fit. This kernel does not. The per-iter
barrier-after-flag F1 disagreement (line 305 `async_write_barrier` instead of flush) is a secondary
concern but the runtime sem id alone is dispositive.

## Action: no edit, ledger status=deferred, flag design-gap.
