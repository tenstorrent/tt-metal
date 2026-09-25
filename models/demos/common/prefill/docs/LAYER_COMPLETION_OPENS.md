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

## 5. Verification debt

Nothing in this area has been built or run on hardware. The C++ is
`-fsyntax-only` clean against real headers; the Python unit tests
(`test_layer_completion_sink.py`, `test_layer_completion_drainer.py`) pass and
are ttnn-free. Unexercised: `LayerAckService::reader_loop`, the drop detector,
D2H + protocol 2 end to end, and both gtests.

`build_Release` is stale (undefined `resolve_bindings`; `compile_commands.json`
has 282 entries). Sync submodules before drawing any conclusion from a build.
