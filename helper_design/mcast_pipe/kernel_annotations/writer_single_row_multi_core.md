# writer_single_row_multi_core.cpp (sort)

Path: `ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_single_row_multi_core.cpp`
Role: **worker confirmation emitter** (writer side). Iterated over `Ht` and bitonic sub-stages.
API era: legacy free-function.

## Block instance — confirmation emit — lines 100-102
- After writing the compare-exchanged tiles back to DRAM (each guarded by `noc_async_write_barrier`, L72/L80/L89/L97):
- `noc_semaphore_inc(coordinator_core_addr, 1)` (L101): increment the coordinator's confirmation counter (the `Wt/2` count the coordinator's Block B waits on).
- `noc_async_atomic_barrier()` (L102): flush the atomic inc.

## Mapping to Pipe
- This is the **return leg** of the handshake: the worker tells the coordinator "I finished this sub-stage." It is the counter-`up()` companion to the coordinator's `wait(number_of_confirmations); reset`.
- No mcast here — pure unicast atomic inc. By itself this is **not the block** (no mcast cluster); it is one half of the cross-kernel sort handshake whose mcast lives in the coordinator.

## Forks
- F1: **atomic barrier** for the inc (L102); **write barrier** for the data (L72/80/89/97). Mixed within one kernel — both fork members coexist.
- F2: counter contribution (inc by 1 per processed pair).
- F3 / pre_handshake: n/a (no mcast, no reset here).

## Note
Annotated for completeness of the sort handshake; in the audit this is **refactor (low)** only as the confirmation-emit companion of the coordinator's receive-side counter. Its mcast-block membership is via its coordinator peer, not standalone.
