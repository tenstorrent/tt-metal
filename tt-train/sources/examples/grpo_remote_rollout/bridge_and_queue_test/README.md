# bridge_and_queue_test

Standalone 2-rank tt-run tests for the two threaded MPI channels used by the
fully-async GRPO trainer: `ThreadedWeightBridge` (device tensors, weight
sync) and `RolloutQueue` (host tensors, rollout batches).

Each channel owns its own duplicated MPI communicator via
`ttnn.distributed_context_duplicate()` so they progress independently under
`MPI_THREAD_MULTIPLE`. Rank assignments match the real trainer:

- **Rank 0 (TTML)**: weight bridge sender, rollout queue consumer.
- **Rank 1 (TTT)** : weight bridge receiver, rollout queue producer.

## Files

- `weight_bridge.py` — `ThreadedWeightBridge` class. On-device pre-allocated
  send/recv pad, event-based cross-CQ ordering, MPI held under the pad lock.
- `rollout_queue.py` — `RolloutQueue` + `RolloutBatch` class. Host-only
  `queue.Queue(maxsize=capacity)` on each side plus a transport thread that
  serializes with `torch.save` and MPI-sends `[u64 length][blob]`. Policy A
  back-pressure (producer blocks when full, no batches lost).
- `test_threaded_bridge.py`, `runner.sh` — bridge-only test.
- `test_rollout_queue.py`, `runner_rollout_queue.sh` — queue-only test.
- `test_bridge_and_queue.py`, `runner_bridge_and_queue.sh` — combined test.
- `configurations/split_1_1/` — shared tt-run config (two [1, 1] Blackhole
  meshes, one no-op fabric intermesh connection). Every test uses this
  same config.

## The three tests

### 1. `test_threaded_bridge.py` (bridge only)

Runner: `./runner.sh`

Rank 0 opens a [1, 1] mesh with `num_command_queues=2`, allocates a live
`ttnn.Tensor`, and every second does 1000 `ttnn.add` calls on CQ0 (mock
gradient work). Every 5 seconds it pushes the tensor through the bridge.
The bridge's sender thread D->H's the pad on CQ1 and MPI-sends the bytes.
Rank 1 receives, H->D's into its own pad on CQ1, and rank 1 main thread
samples the pad on CQ0. Rank 0 ships the expected first-element trail on
the world context so rank 1 can print `[PASS]` / `[FAIL]`.

Verifies:

- End-to-end round-trip of a live device tensor.
- CQ0 (main) and CQ1 (bridge) run in parallel without stalling each other.
- Event-based cross-CQ ordering (`ttnn.record_event` /
  `ttnn.wait_for_event`) instead of `synchronize_device`.

### 2. `test_rollout_queue.py` (queue only)

Runner: `./runner_rollout_queue.sh`

Rank 1 builds `N_BATCHES = 10` deterministic `RolloutBatch` objects
(deterministic ragged token lists plus a fp32 log-prob tensor) and pushes
each one through the queue. Rank 0 pops until `pop()` returns `None`,
computes the same checksum for each received batch, and compares against
the expected checksums shipped separately on the world context.

The consumer sleeps 250 ms per pop and the local queue has
`capacity=2`, so after 3+ batches are in flight the producer's `push()`
wall-clock time visibly grows -- back-pressure confirmed in the log.

Verifies:

- Round-trip of host-side batches with variable-length tokens and a
  fixed-shape fp32 log-prob tensor.
- Policy A back-pressure from consumer to producer via the local queue +
  MPI wire.
- The close message (`length == 0`) drains the pipeline cleanly.

### 3. `test_bridge_and_queue.py` (both together)

Runner: `./runner_bridge_and_queue.sh`

Both channels connected at the same time, each on its own duplicated MPI
context. Both ranks call `bridge.connect()` and then `queue.connect()`
in matching order.

Rank 0 main loop (`N_ROUNDS = 6`), mirroring the training loop:

1. `queue.pop()` -- get a rollout batch from rank 1.
2. `ttnn.add(...)` for `ADDS_PER_ROUND` iterations -- mock gradient work.
3. Every other round, `bridge.push_tensor(x)` -- ship the updated weight.

Rank 1 main loop:

1. `queue.push(batch)` -- send a fresh fake rollout batch.
2. Peek the bridge's recv pad and, if a fresh weight arrived, sample +
   verify + release.

At the end, rank 1 ships two summary lists (rollout checksums + observed
weight first-elems) to rank 0, which prints a PASS/FAIL per channel plus
a top-level `[COMBINED PASS]` / `[COMBINED FAIL]`.

Verifies:

- Both duplicated MPI contexts progress independently under
  `MPI_THREAD_MULTIPLE`.
- Trainer-style loop ordering (consume rollout -> compute -> publish
  weights) works end to end.
- If the ttml `DistributedContext` bindings ever regress and stop
  releasing the GIL, this test deadlocks in a way the two solo tests
  might miss.

## Running

Prereqs: `TT_METAL_HOME` set, the ttnn/ttml Python venv activated. All
runners take the same optional flags (`--hostfile`, `--rank-bindings`,
`--script`) with the same defaults from `configurations/split_1_1/`.

```
cd $TT_METAL_HOME
./tt-train/sources/examples/grpo_remote_rollout/bridge_and_queue_test/runner.sh
./tt-train/sources/examples/grpo_remote_rollout/bridge_and_queue_test/runner_rollout_queue.sh
./tt-train/sources/examples/grpo_remote_rollout/bridge_and_queue_test/runner_bridge_and_queue.sh
```

Look for `[PASS]` (individual tests) or `[COMBINED PASS]` (combined test)
in the tail of the output.
