# Annotation — `writer.cpp` (conv3d)

Path: `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/`
Role: **HYBRID (sender + receiver), THREE weight-share modes** selected at compile time:
`off (0)` / `chain (1)` / `mcast (2)` — `WeightShareRole` enum in `conv3d_weight_share.hpp:24`.
Substrate: new object API.

> This file has TWO distinct sync channels:
> 1. **mcast channel** (mode 2) — the true `Pipe` block. F3 = **EXCLUDE_SRC**.
> 2. **unicast chain channel** (mode 1) — a peer-to-peer forwarding chain using `async_write` +
>    `async_write_barrier` + remote `up` + `async_atomic_barrier`. **NOT multicast.** It is a
>    *sibling* of the `Pipe` problem (same handshake hazards, different transport). Flagged as a
>    near-miss the recognition family does NOT spell — see return.

---

## Variant signature (mcast mode, `L197-253`)

| Fork | Value | Lines |
|---|---|---|
| **F1** flush vs barrier | **NEITHER** on the mcast path (comment `L232-237` cites the conv2d VC-4 FIFO reasoning = INV4). | (none on mcast) |
| **F2** flag vs counter | **flag.** R→S: sender `wait(mcast_num_dests)` exact `L210` + `set(0)` `L211`. S→R: `set_multicast(VALID)` `L239`. Receiver: `up` `L249` → `wait(1)` `L250` → **own `set(0)` reset `L251`**. | 210,211,238,239,249,250,251 |
| **F3** loopback | **EXCLUDE_SRC** (`L219`, `L239`). Comment `L213-215`: sender keeps its DRAM-read copy, so no loopback; `mcast_num_dests = bbox_cores - 1`. | 219,239 |
| **KNOB** pre_handshake | **YES (dest reused)** — sender pre-acks-waits `L210`. | 210 |

> **F2 NUANCE vs conv2d:** here the **RECEIVER resets its OWN flag** (`set(0)` at `L251`, also
> `L143`/`L175`) AFTER `wait(1)`, rather than the conv2d pattern where the receiver pre-arms
> `set(INVALID)` BEFORE acking and the sender re-stages VALID each round. conv3d also uses
> `wait(1)`/`set(0)` (count-to-1) rather than VALID/INVALID named levels — semantically the same
> level-flag-with-reset, different spelling. **A spelling the `Pipe` must subsume.**

---

## mcast mode block — SENDER (`McastSender`, `L199-246`)
- `L204-205` `read_weight_block(...)` — DRAM read into local `cb_weight` (source L1 fill).
- `L210` `weights_mcast_sender_sem.wait(mcast_num_dests)` — **R→S pre-handshake** (H2/KNOB, exact
  `wait` ⇒ F2 flag). Comment `L207-209` documents `mcast_num_dests` = receivers = acks = EXCLUDE_SRC
  `num_dests` (H8 accounting, host places sender in bbox so = bbox_cores-1).
- `L211` `set(0)` — reset ack counter (H3).
- `L216-217` `weight_block_bytes`, `local_addr = cb_weight.get_write_ptr()` (ANY src/dst addr).
- `L219-230` `async_write_multicast<EXCLUDE_SRC>(...)`, `linked=true`. **DATA mcast — enqueued.**
  F3 = M7a (sender keeps DRAM copy). bbox = `MulticastEndpoint` rect.
- `L238` `weights_mcast_receiver_sem.set(VALID)` — stage level (A5).
- `L239-246` `set_multicast<EXCLUDE_SRC>(VALID)` — **FLAG mcast**, `linked=false`. INV4: after data,
  same Noc/VC, **no barrier** (comment `L232-237` is the most explicit INV4 rationale in the group).
- `cb_weight.push_back(weight_tiles)` `L253`.

## mcast mode block — RECEIVER (`McastReceiver`, `L247-252`)
- `L249` `weights_mcast_sender_sem.up(noc, weight_src_noc_x, weight_src_noc_y, 1)` — **R→S ack** (A8).
- `L250` `weights_mcast_receiver_sem.wait(1)` — **S→R wait** (A6, threshold 1). Data landed by INV4.
- `L251` `weights_mcast_receiver_sem.set(0)` — **own-flag reset AFTER wait** (H3, receiver-side reset
  variant). Then `push_back`.

## Out-of-block channel 1 — UNICAST CHAIN (mode 1, `L162-196`) — NOT mcast
- `L169-171` injector: DRAM read (chain head).
- `L173-176` non-injector receiver half: `up` ack → `wait(1)` → `set(0)` (same level-flag-with-reset
  as mcast receiver, but the data arrives by **unicast**, not mcast).
- `L177-194` forwarder (non-tail): `wait(1)`+`set(0)` for upstream readiness, then `async_write`
  (unicast to chain successor `L184-189`), `async_write_barrier` `L190` (**F1=barrier — required, no
  flag-mcast/VC-FIFO to lean on**), then `up` successor `L192` + `async_atomic_barrier` `L193`.
- **HOLE w.r.t. the mcast `Pipe`:** this is a unicast forwarding chain. Same hazard family (H1
  source-clobber → barrier; H3 flag reset) but a **different transport** (unicast + per-hop barrier).
  The `Pipe` as scoped (multicast) does NOT cover it. Note it: a future `Pipe` could expose a
  unicast-chain mode, but that is a scope expansion, not this run.

## Out-of-block channel 2 — DRAIN loop (`L140-146`) — receiver-only ack
- Cores not participating in compute still must ack the sender each iter so the sender's
  `wait(mcast_num_dests)` completes: `up` `L141` → `wait(1)` `L142` → `set(0)` `L143`, ×`mcast_num_iters`,
  then `async_atomic_barrier` `L145`. This is the **receiver half with no consume** — a degenerate
  `Pipe::receive` (ack+wait+reset, drop the data). Relevant: the `Pipe::receive` must support a
  "drain / ack-only" mode where no CB slot is consumed. → catalog amendment candidate.
