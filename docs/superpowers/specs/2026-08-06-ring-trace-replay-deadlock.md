# Ring-attention trace-replay deadlock — root cause

Traced chunked prefill (gemma4, 4x8 Blackhole galaxy, CP=4 along mesh axis 0) deadlocks
at deep ring depth. Observed at chunks 54, 59 and 61 of 64 in the real test, and at
replays 41–61 in the stress reproducer. It is probabilistic: the same configuration
sometimes completes. Recovery requires a board reset.

## Where it is stuck

From `tt-triage --run=dump_callstacks` taken on a live hang (inspector enabled, triage
attached before killing the process):

| kernel | stuck at |
|---|---|
| `ring_attention_all_gather_reader` | `noc_semaphore_wait_min` — **reader.cpp:201** |
| `ring_attention_all_gather_writer` | `cb_wait_front` (waiting on its own local reader) |
| `ring_joint_reader` / `ring_joint_sdpa` | `RingSDPAOpReceiver::get_next_ring_id_and_consume_one_signal` → `Semaphore::down(1)` |

Line 201 is `noc_semaphore_wait_min(out_ready_sem, slices_received + 1)`.

## The decisive measurement

`out_ready_sem` values read out of L1 with ttexalens while deadlocked, laid out by CP
ring (mesh rows = CP axis, columns = TP):

```
                     c0  c1  c2  c3  c4  c5  c6  c7
 row0 (ring endpoint) 0   0   0   0   0   0   0   0     <- COMPLETED, zeroed at kernel exit
 row1 (ring middle)   2   2   2   2   1   1   1   1     <- STUCK, one increment short
 row2 (ring middle)   2   2   2   2   1   1   1   1     <- STUCK, one increment short
 row3 (ring endpoint) 0   0   0   0   0   0   0   0     <- COMPLETED, zeroed at kernel exit
```

Reproduced **identically** in two independent captures, so this is a deterministic
deadlock shape, not random corruption. Every ring is in the same state: both endpoints
finished and reset their semaphore, both middles wait for an increment whose sender has
already exited. The two middles then wait on each other — neither writer can forward,
because neither reader has received.

## Root cause

Two counting semaphores span trace replays, and neither is re-initialized by a replay
(the program is not recreated):

1. `out_ready_sem` — a `GlobalSemaphore`, incremented **remotely over the fabric** by the
   neighbour's writer, and **destructively reset** by the local reader at kernel exit:
   `noc_semaphore_set(out_ready_sem, 0)` (reader.cpp:281).
2. The fused-op signal semaphore — incremented by the all-gather worker, consumed by the
   SDPA reader with `Semaphore::down(1)` (fused_op_receiver.hpp:52). Created once at 0 and
   never resynchronized, so any producer/consumer imbalance is **permanent**.

The increments are asynchronous. The writer's `send_payload_flush_blocking_from_address`
flushes into the local EDM only — the sender's kernel completes while its increment is
still traversing ethernet. Nothing establishes that every increment belonging to
invocation N has been delivered before the receiving reader exits and zeroes the
semaphore. When one lands after that reset it is destroyed, the counts drift by one, and
the drift never heals — so a later invocation waits for an increment that will never come.

Deep ring depth matters because the gather grows with `kv_actual_isl` (195 ms → 512 ms per
chunk), which widens inter-device skew and therefore the window.

## Evidence table

| experiment | result | rules out / establishes |
|---|---|---|
| fixed depth, 120 replays | survived | not replay-count accumulation |
| shallow depth (chunks 1–10), metadata changing every replay | survived 120 | not stale-metadata reads |
| deep-only (chunks 40–62), 400 replays | survived | depth alone is not sufficient |
| full sweep 1–63 | hangs ~replay 41–61 (sometimes survives) | needs the full progression; probabilistic |
| + 200 ms / 25 ms sleep between replays | survived | wall-clock drain prevents it |
| + 5 ms sleep | hangs | drain must exceed the in-flight window |
| `ttnn.synchronize_device` after staging | **does not fix** | the device barrier does not cover in-flight inter-chip traffic |
| host reset of the ring semaphores between replays | does not reliably fix | resetting cannot beat an arrival that lands after it |
| `num_links=1` | still hangs | not the two parallel link workers racing |
| `readback_all` (eager all_gather + host read per chunk) | never hangs | supplies the same drain incidentally |

## Why it surfaced now

`readback_all` reads every chunk's hidden states to host, which runs an eager
`ttnn.all_gather` and a device→host read between replays. That incidentally drained the
fabric. Removing that test-only readback — the correct thing to do for a production-shaped
measurement — removed the accidental synchronization and exposed the latent bug. Eager
execution never hits it because per-op host dispatch supplies far more slack.

## Fix direction (op level)

The workaround in the test is a 25 ms settle between replays
(`GEMMA4_TRACE_SETTLE_MS`), costing ~1.6 s of a 24.7 s 256k prefill. The real fix belongs
in the all-gather:

- **Preferred:** stop destructively resetting. Carry a per-core base across invocations and
  wait on `base + k`, so a late arrival is absorbed by the next invocation instead of being
  destroyed. Needs no extra synchronization and no round trip.
- Alternative: have the receiver acknowledge completion to the sender before resetting
  (adds a round trip per invocation).
- Alternative: make the trace-replay boundary a fabric quiescence point, which is what the
  sleep approximates.

Whatever the choice, the fused-op signal semaphore should also be resynchronized per
invocation rather than relying on producer/consumer counts never drifting.

## Reproducer

`test_prefill_trace_replay_stress` in `models/demos/gemma4/demo/text_demo_prefill.py`
replays one captured trace repeatedly and separates the variables above:

```bash
GEMMA4_STRESS_REPLAYS=400 pytest \
  "models/demos/gemma4/demo/text_demo_prefill.py::test_prefill_trace_replay_stress" \
  -k advancing_depth -s -v
```

`GEMMA4_STRESS_MINCHUNK` / `MAXCHUNK` bound the depth sweep, `GEMMA4_STRESS_DELAY_MS` adds
a settle, `GEMMA4_STRESS_SYNC_STAGE=1` adds a device barrier. To capture state, run with
`TT_METAL_INSPECTOR=1` and, while it is still hung, attach:

```bash
./tools/tt-triage.py --run=dump_callstacks --llm-output --llm-output-path=/tmp/triage.txt
```

Do not kill the process first — the state is gone once it dies. Note triage reads inspector
logs from `<repo>/generated/inspector`, not the `/tmp/tt-metal/inspector` default.
