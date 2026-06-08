# reader_mcast_sender_unary_sharded_ln.cpp (SEND side)

Path: ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp

API spelling: **experimental OO wrapper** — `Noc`, `Semaphore<>`, `MulticastEndpoint`/`UnicastEndpoint`, `CircularBuffer`. Maps to raw `noc_semaphore_set/set_multicast`, `noc_async_write_multicast`, `noc_async_*_barrier`. NOT the bare `noc_*` spelling.

Role: coordinator/sender for split-across-cores reduce (mean+var). One-stage or two-stage. Block runs up to twice (mean lambda L290, var lambda L294).

## Block map (inside `global_reduce_sender` lambda L106-287)

### Phase-1 handshake (R→S gather signal), COUNTER style + level flag, EXCLUDE_SRC
- L122 `cb_partial_obj.wait_front(...)` — wait local partial ready (producer = compute).
- L124-135 `if num_blocks>1`:
  - L125 `reduce_sender_sem.set(VALID)` — arm the level flag locally.
  - L126 `reduce_receiver_sem.wait(num_blocks-1)` — **COUNTER wait**: wait for N-1 receivers to inc.
  - L127 `reduce_receiver_sem.set(0)` — **counter reset** (reused slot across the two lambda invocations / next op iter).
  - L128-134 `reduce_sender_sem.set_multicast(... num_blocks-1)` — mcast the level flag VALID to the rectangle (EXCLUDE_SRC default). This releases receivers to read.
- **F2 = HYBRID here**: phase-1 uses BOTH a monotone-ish counter (`reduce_receiver_sem`, wait+reset) AND a level flag (`reduce_sender_sem` set VALID / mcast). This is the layernorm "two-phase" signature noted in the brief.

### Gather (NoC reads, not part of Pipe block) L148-213
- Remote `noc.async_read` loops + `async_read_barrier` (L166, L175, L191, L203). These are unicast reads of partials, NOT the mcast block; HOLE for Pipe (the helper wraps the mcast+handshake, not the gather reads).

### Phase-2 pre-mcast handshake (gather-done), COUNTER, EXCLUDE_SRC
- L212 `cb_ex_obj.wait_front(...)` — wait combined result ready (producer = compute).
- L214-217 `if num_all_to_all_workers_first_stage>1`: `reduce_receiver_sem.wait(N-1)` then `.set(0)` — second COUNTER wait+reset, gating before mcast of finals.

### Phase-2 streaming mcast (data + flag), per-block loop, COUNTER promotion, EXCLUDE_SRC
- L225 `cb_ex_global_obj.reserve_back(...)` — fresh dest slot.
- L230-250 gather finals into cb_ex_global (NoC reads, HOLE).
- L253 `cb_ex_global_obj.push_back(...)`.
- L254-286 `if num_blocks>1` per-block loop:
  - **L257 `reduce_sender_sem.set(block+2)`** — **MONOTONE COUNTER**: the flag is set to an increasing value (2,3,4,...) so each receiver's `wait_min(block+2)` (see receiver) advances one streamed block. This is the **counter-phase-2** half of the two-phase pattern.
  - L263-274 `noc.async_write_multicast(cb_ex_global_obj, mcast_ep, ... num_blocks-1, ..., true)` — **DATA mcast**, linked=true. EXCLUDE_SRC (default).
  - L275-281 `reduce_sender_sem.set_multicast(... num_blocks-1)` — mcast the counter value to the rectangle.
  - L284 `noc.async_write_barrier()` — **F1 = BARRIER** (per streamed block).

## Variant signature
- **F1 = barrier** (L284, `async_write_barrier` per block; also the gather uses read barriers).
- **F2 = phase-1 flag+counter hybrid; phase-2 MONOTONE COUNTER** (`set(block+2)` + receiver `wait_min`). Both styles present in one kernel.
- **F3 = EXCLUDE_SRC** (sender is coordinator; default mcast mode; sender self-fills via its own gather, not loopback).
- **pre_handshake**: phase-2 mcast IS preceded by a R→S handshake (L214-217 counter wait), so dest is effectively re-synced. But cb_ex_global is a **fresh reserve_back slot** (L225) each op-iter — so the "fresh CB slot" knob applies to the *data*, while the *handshake* still gates. NOT the no-pre-handshake case.

## Hazards / invariants
- INV: `reduce_sender_sem` flag VALID must be mcast AFTER local partials are read-stable (sender reads its own cb_partial). Receivers gated on the flag.
- HAZARD: counter reset `reduce_receiver_sem.set(0)` (L127, L216) must happen after the wait completes and before the next phase/iteration re-uses the slot — reused slot, not fresh.
- HAZARD (streaming): `set(block+2)` monotone values rely on receiver `wait_min`; if reset between op-iters is wrong the monotone base drifts. The base is re-established by L257 starting at block=0 → value 2 each call (because the lambda re-enters and the receiver also resets). Cross-iter correctness depends on receiver consuming all `block+2` levels before sender re-enters.
- HOLE: the interleaved NoC gather reads (L162-213, L230-250) are interleaved with the handshake; a Pipe `send()` that only wraps mcast+flag cannot subsume them — they must stay at the call site between Pipe phases.
