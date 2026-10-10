<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Layer completion — open design decisions

Scratchpad for the decisions still outstanding on the layer-completion path
(issue #54632). Scope is **surfacing the completion signal only** — what
consumes it (migration, scheduling) is decided elsewhere and is deliberately
not argued here.

Code entry points:

- `runners/layer_completion_sink.py` — producer side, v1 count / v2 structured
- `runners/layer_completion_drainer.py` — consumer side, coverage accounting
- `runners/prefill_runner.py` (`_serve_request`) — transport wiring
- `ttnn/core/services/layer_ack_service.cpp` — D2H transport
- `models/demos/deepseek_v3_d_p/tt/kv_ack.py` — where the ack actually fires

---

## 1. Dense layer acks (supersedes #2 if it lands)

**Today** the ack is gated on `attention.writes_kv` (`kimi_k3/block.py:209`),
because it is sited inside `zero_pad_and_ack`. So "layer completed" inherited
"layer wrote a KV slab" as its definition, and a 24-layer Kimi-K3 rank emits 6
completions, not 24.

**Proposal** — hoist the ack out of the `writes_kv` branch (the pad-zero stays
in). A completion means "layer N of request R is done", nothing more, so every
layer has one.

**Why it matters** — the sparse ack is the sole reason all of this exists:

- `ack_idx_of_layer` / `num_ack_layers` / `ack_first_idx` / `ack_local_count`
  in `prefill_runner.py`
- the `ack_layer_ids` mapping on `LayerAckService`
- the zero-ack-rank `RuntimeError` (unreachable once every rank acks)
- `NUM_ACK_LAYERS` in `prefill_producer.py`
- **open #2 below, entirely** — dense spans tile `[0, num_layers)` by
  construction

**What blocks it** — cost, and it is a property of the transport, not of the
semantics:

| transport | cost per ack | dense (Kimi-K3, 24 vs 6) |
|---|---|---|
| host callback, untraced | `ttnn.synchronize_device` (`kv_ack.py:127`) | 6 → 24 syncs/chunk |
| host callback, traced | trace split + sync at each `_ACK` (`sub_device_trace.py` `replay()`) | 6 → 24 segments + syncs/chunk |
| D2H | device op on the same CQ, **no host sync** (`kv_ack.py:62-66`) | free; 4 KB FIFO holds ~341 × 12 B records |

`replay()`'s own docstring notes the per-ack sync deliberately replaced
"block-every-segment (which serialized dispatch for no reason but the ack)" —
so dense acks on the host path undo a tuned optimization.

**Sequencing** — dense acks are cheap exactly where open #3 is heading.
Dense-before-D2H-everywhere regresses hybrid models on the host path;
dense-after costs nothing. Suggested: hoist behind `PREFILL_DENSE_LAYER_ACK`
(default off), measure both transports on hardware, then flip the default and
delete the ACK-space machinery in the same commit.

## 2. v2 span policy on a hybrid stack (moot if #1 lands)

Both transports emit the raw global span `[l, l+1)` for the acking layer. On a
hybrid stack those are sparse (`{3,7,11,15,19,23}` on a 24-layer Kimi-K3 rank),
so `RequestCoverage.layers_accounted` tops out at 6/24, `is_complete()` never
returns true, and `on_request_complete` never fires.

Reproduced: `drain_blocking` itself terminates correctly (span-sum reaches
`NUM_ACK_LAYERS`); it is the **completion signal** that is missing, not the
drain. Latent today — nothing in production passes `on_request_complete` or
`on_first_completion`; it bites the first real v2 consumer.

Options, if #1 does not land:

- **(a)** emit the span in ACK space — coverage tiles, but `layer_start/end`
  then addresses a KV slot rather than a layer
- **(b)** keep the span global and make the drainer's `is_complete` ACK-aware
- **(c)** widen the span to what the ack attests — `[0,4) [4,8) …` — dense in
  GLOBAL space, no translation, and the case the range-native interface was
  built for

`TODO(#54632)` markers in `layer_completion_sink.py` and
`layer_ack_service.cpp` point here.

## 3. D2H on every model

`PREFILL_LAYER_ACK_D2H` defaults off. MiniMax-M3 and GPT-OSS raise
`NotImplementedError` on a `d2h_service`: they have **no device-side ack op at
all**, only a plain callback in the Python forward loop
(`minimax_m3/tt/model.py:386`, `gpt_oss_d_p/tt/model.py:210`). Making D2H
universal needs a device ack site added to each model's forward, sited against
its KV writes. That is per-model work, and it is where the schedule risk sits —
not in the transport.

The host-callback path is **deliberately retained**, not a fallback: it is the
only path those two models have, it is the reference semantics for `layer_idx`
and the v2 span (true global layer, not reconstructed from a counter), and it
needs no trace coupling. See the note at the `else:` branch in
`prefill_runner.py`.

## 4. `request_id` is still counter-derived on D2H

`LayerAckService` now reads the record's `{slot_id, actual_start, actual_end}`
(it used to discard them) and reconstructs the true global `layer_idx` from
`ack_layer_ids`. `request_id` remains `k / local_layers`. Options: stamp a 4th
word into the metadata upstream (the producer already packs it; the D2H
service's `metadata_size_bytes` is an independent parameter and is already
aligned up, so there may be free space), or key the consumer on
`(slot_id, pos_start, pos_end)`, which is already a unique chunk identity that
`RequestCoverage.record_identity` validates.

Mitigated meanwhile: a chunk boundary that does not land on ack index 0 means
records were dropped, and `LayerAckService` now logs that instead of silently
relabelling everything after it.

## 5. Verification status (2026-10-10)

Run on Blackhole galaxies (bh-glx-110-a10u02/08/14/20): protocol 2 end to end with the tt-d-gen
prefill reader on GLM-5.3 (4 ranks, D2H), Kimi K2.7 (4 ranks, D2H) and MiniMax-M3 (1 rank, host
callback); protocol 1 on the same; the D2H record ring and the `LayerAckService` device tests
(`tests/ttnn/unit_tests/base_functionality/test_layer_ack_service.py`); the host-only gtests
(`distributed_unit_tests --gtest_filter='LayerCompletion*:LayerAck*'`, on a galaxy host: the test
binary opens the devices at startup) and the Python unit tests. The review and cleanup record is
`LAYER_COMPLETION_REVIEW_PLAN.md`.

### Hardware legs (2026-10-10, bh-glx-110-a10u02)

DeepSeek-R1 1-rank, D2H, lcv2 worker: I1 eager and traced PASS; N3 (SIGSTOP 8 s stock, 25 s with `ETCD_LEASE_TTL=120`)
PASS; N2 SIGINT PASS; N2 SIGKILL is not survivable end to end (torn H2D request, runner assert — the request path, not
this protocol) and exposed two tt-d-gen gaps fixed in its lcv2 tree (stray completions dropped and counted; a dead
engine thread fails liveness). N1 both ways PASS; a protocol-2 worker on a published protocol-1 counter segment now
fails in 2 s instead of the connect budget (`LayerCompletionQueue::connect` grace, db87d721778). The 2-rank GLM-5.3 and
Kimi K2.7 legs were blocked when the a10u14 allocation ended mid-run. Full results: LAYER_COMPLETION_REVIEW_PLAN.md §7.

## 6. Scheduler segment names

`InterProcessCounterChannel::connect` validates nothing about the segment it attaches to, so the
two protocols cannot share a name: a protocol-1 consumer on a protocol-2 ring reads `enqueue_pos`
as its count and writes its cursor into `dequeue_pos`. v1 stays at `/tt_prefill_layer_acks_<svc>`
(existing consumers); v2 publishes `/tt_prefill_layer_completions_<svc>`; `scheduler_shm_name()`
in `layer_completion_drainer.py` is the one place that knows both. tt-d-gen's `ack_shm_name`
configs follow the same rule.

## 7. Hybrid stacks under protocol 2

The v2 span is the raw global layer `[l, l+1)` (open #2). The tt-d-gen reader retires a chunk when
`[0, layers_per_chunk)` is covered and fatals on a layer past that bound, so a 24-layer Kimi-K3 rank
with six acks never retires under protocol 2. Hybrid models stay on protocol 1 until open #2 (or
#1) is decided; the dense models run both.
