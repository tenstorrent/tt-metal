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

- `weight_bridge.py` -- `ThreadedWeightBridge` class. Public API mirrors
  the `WeightBridge` ABC in `utils/weight_bridge.py`:
  `send_weights(dict) / receive_weights() (context manager) /
  poll_weights() (context manager) / connect() / close()`. On-device
  pads are lazy-allocated one per key from the first send / first
  received manifest.
- `rollout_queue.py` -- `RolloutQueue` + `RolloutBatch` class. Host-only
  `queue.Queue(maxsize=capacity)` on each side plus a transport thread
  that serializes with `torch.save` and MPI-sends `[u64 length][blob]`.
  Policy A back-pressure (producer blocks when full, no batches lost).
- `test_threaded_bridge.py`, `runner.sh` -- bridge-only test.
- `test_rollout_queue.py`, `runner_rollout_queue.sh` -- queue-only test.
- `test_bridge_and_queue.py`, `runner_bridge_and_queue.sh` -- combined
  test.
- `configurations/split_1_1/` -- shared tt-run config (two [1, 1]
  Blackhole meshes, one no-op fabric intermesh connection). Every test
  uses this same config.

## The bridge's API in one screen

```python
bridge = ThreadedWeightBridge.sender(peer_rank=1, mesh_device=mesh)
bridge.connect()
bridge.send_weights({"w0": x0, "w1": x1})   # caller can now mutate x0/x1
bridge.close()

bridge = ThreadedWeightBridge.receiver(peer_rank=0, mesh_device=mesh)
bridge.connect()
with bridge.receive_weights() as dicts:      # blocks; lock held here
    for k, t in dicts[0].items():
        sample = float(ttnn.to_torch(t, cq_id=0)[0, 0])
        ...                                  # bridge cannot overwrite
# lock released; bridge free to advance to next incoming dict
with bridge.poll_weights() as dicts:         # non-blocking
    if dicts is None:
        pass                                 # nothing ready
    else:
        ...                                  # same read pattern as above
bridge.close()
```

The freeze point on the send side is the `ttnn.copy(src, pad, queue_id=0)`
inside `send_weights`; the caller is free to keep mutating its live
source tensors immediately after `send_weights` returns. The freeze point
on the receive side is the recv pad lock held across the `with` block;
the bridge cannot overwrite the pads until the `with` body exits.

## The three tests

### 1. `test_threaded_bridge.py` (bridge only)

Runner: `./runner.sh`

Rank 0 opens a [1, 1] mesh with `num_command_queues=2`, allocates two
live fp32 tensors `x0` (grows +1 per add) and `x1` (grows +2 per add),
and every second does 1000 `ttnn.add` calls on CQ0. Every 5 seconds it
calls `bridge.send_weights({"w0": x0, "w1": x1})` AND THEN does another
1000-add burst on the same `x0` / `x1` -- explicitly demonstrating that
the caller keeps mutating its live tensors after `send_weights` returns.

The bridge's sender thread sends a manifest, then per-key
`ttnn.to_torch(cq_id=1)` + `torch.save` + MPI-send. Rank 1 uses
`with bridge.receive_weights() as dicts:`; the recv pad lock is held
across the `with` body so the bridge cannot overwrite the recv pads
while rank 1 reads them. Rank 1 samples the first elem of each key on
CQ0 and compares to the expected trail shipped by rank 0 on the world
context. Prints `[PASS]` / `[FAIL]`.

Verifies:

- End-to-end round-trip of a multi-key on-device
  `dict[str, ttnn.Tensor]`, with lazy-init on-device pads.
- Caller-side mutation-freedom: rank 0 mutates `x0` / `x1` right after
  `send_weights` returns; the pads are the freeze point.
- Recv pad lock held across the `with` block: the bridge cannot corrupt
  the caller's read.
- Manifest + per-tensor blob wire format (mirrors `HostWeightBridge`).
- CQ0 (main thread `ttnn.add` bursts) and CQ1 (bridge thread `to_torch`)
  run concurrently without stalling each other.
- Close handshake (length-0 manifest = close message).

### 2. `test_rollout_queue.py` (queue only)

Runner: `./runner_rollout_queue.sh`

Rank 1 builds `N_BATCHES = 10` deterministic `RolloutBatch` objects and
pushes each one through the queue. Rank 0 pops until `pop()` returns
`None`, computes the same checksum per batch, and compares against the
expected checksums shipped separately on the world context.

The consumer sleeps 250 ms per pop and the local queue has
`capacity=2`, so after 3+ batches are in flight the producer's `push()`
wall-clock time visibly grows -- back-pressure confirmed in the log.

Verifies:

- Round-trip of host-side batches with variable-length tokens and a
  fixed-shape fp32 log-prob tensor.
- Policy A back-pressure from consumer to producer via the local queue
  + MPI wire.
- The close message (`length == 0`) drains the pipeline cleanly.

### 3. `test_bridge_and_queue.py` (both together)

Runner: `./runner_bridge_and_queue.sh`

Both channels connected at the same time, each on its own duplicated
MPI context. Both ranks call `bridge.connect()` then `queue.connect()`
in matching order.

Rank 0 main loop (`N_ROUNDS = 6`), mirroring the training loop:

1. `queue.pop()` -- get a rollout batch from rank 1.
2. `ttnn.add(...)` on `x0` and `x1` -- mock gradient work.
3. Every other round, `bridge.send_weights({"w0": x0, "w1": x1})` and
   then keep mutating x0 / x1 (mutation-freedom property, same as test
   1).

Rank 1 main loop:

1. `queue.push(batch)` -- send a fresh fake rollout batch.
2. `with bridge.poll_weights() as dicts:` non-blocking. If a fresh dict
   is there (dicts is not None), sample first elem of each key under
   the lock and record.

At the end, rank 1 ships two summary lists (rollout checksums +
observed weight first-elem dicts per key) to rank 0, which prints a
PASS/FAIL per channel plus a top-level `[COMBINED PASS]` /
`[COMBINED FAIL]`.

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

Look for `[PASS]` (individual tests) or `[COMBINED PASS]` (combined
test) in the tail of the output.
