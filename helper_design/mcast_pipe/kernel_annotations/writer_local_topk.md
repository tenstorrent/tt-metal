# writer_local_topk.cpp (topk)

Path: `ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_local_topk.cpp`
Role: **local/worker (SENDER) core** of topk aggregation. Waits for the final core's invite, unicasts its local TopK tiles to the final core, then signals completion. Iterated over `Ht`.
API era: **modern** (`Noc`, `Semaphore<>`, `UnicastEndpoint`, `CircularBuffer`).

## Semaphores
- `receiver_sem` (L32): final-core readiness flag, received (set by the coordinator's mcast).
- `sender_sem` (L33): completion counter on the final core, incremented outbound.

## Block instance — per-row send — lines 56-96
- `receiver_sem.wait(VALID)` (L56): wait for the coordinator's mcast invite (RECEIVE the flag).
- data transfer (NOT mcast — unicast to the final core's L1):
  - values: loop `values_cb.wait_front` + `noc.async_write(... noc_final_x/y, final_values_base+...)` + `pop_front` (L60-71).
  - indices: same for indices (L75-86).
- `noc.async_write_barrier()` (L89): flush data writes before signaling.
- `sender_sem.up(noc, noc_final_x, noc_final_y, Kt)` (L92): increment final core's arrival counter by Kt.
- `noc.async_atomic_barrier()` (L93): flush atomic inc.
- `receiver_sem.set(INVALID)` (L96): **reset** local invite flag for next round.
- trailing `noc.async_atomic_barrier()` (L100) once after loop.

## Mapping to Pipe
- The SENDER companion of `reader_final_topk`. `Pipe::send()` here = `wait(invite) ; <push data> ; barrier ; up(counter) ; atomic_barrier ; reset(invite)`.
- Data is unicast (each worker writes to its own offset slice in the final core's CB) — the mcast is only the *invite flag*, owned by the coordinator. So the data-fan-in is unicast scatter, not mcast.

## Forks
- F1: **write barrier** for data (L89) + **atomic barrier** for the inc (L93). Both members coexist.
- F2: inbound invite is a **flag** (`wait(VALID)` / `set(INVALID)` reset); outbound completion is a **counter** (`up(Kt)`). Both members.
- F3: n/a here (no mcast emitted by this kernel; it consumes one).
- KNOB pre_handshake: **fresh slot** — local `receiver_sem` reset to INVALID each row (L96).

## HOLEs
- Reset of the invite flag is split: coordinator resets it on its own copy (`set(VALID)` then mcast), worker resets its *local* copy to INVALID after consuming. Cross-kernel reset ownership hazard (same as sort).
- Per-core L1 dest offset (`final_values_base + start_wt*...`) — the data-scatter target is computed from a runtime `start_wt`; if a Pipe were to also carry the data, it must support per-sender dest offsets.
