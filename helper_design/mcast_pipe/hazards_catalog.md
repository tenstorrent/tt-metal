# Step B — Hazards / Invariants Catalog (`mcast_pipe`, dataflow)

Derived **entirely from the Step A primitive contracts** — reads no kernels. A hazard is a race
the primitives permit; each has ≥1 mitigation. A hazard with **≥2 viable mitigations** is a
**style fork** → goes to the Step E bake-off (unless a mitigation choice is *forced by the
caller's program*, in which case it is a use-case **knob**, not a fork).

The mcast `static_assert`s from Step A (NON_POSTED-only, no-trid, VC-4-only) already pruned the
fork space — POSTED-mcast, TXN_ID-barrier, and custom-VC forks **do not exist** and are not listed.

---

## Hazards

### H1 — Source-L1 clobber (sender overwrites src before it's SENT)
- **Race:** `async_write_multicast`/`set_multicast` return at *enqueued* (A1, A5). The src L1 —
  the data buffer for A1, the **sem cell itself** for A5 — is still being read by the NoC engine.
  If the sender overwrites it (next iter's fill, or `set()` of a new flag value) before the bytes
  are **SENT**, the in-flight payload corrupts.
- **Mitigations (≥2 ⇒ FORK):**
  - **M1a flush** — `async_writes_flushed()` waits for **SENT** (A2). Cheap local poll. Sufficient.
  - **M1b barrier** — `async_write_barrier()` waits for **ACKed** (A3). Also sufficient; strictly
    more expensive (NoC round-trip).
- **FORK F1 = flush vs barrier.** Both correct for H1; the question is the perf gap and whether
  any topology makes barrier *necessary* (see H4 interaction).

### H2 — Destination-L1 overrun (sender mcasts into a dest the receiver hasn't drained)
- **Race:** sender mcasts iter *i+1* into the **same dest L1** before receivers consumed iter *i*
  → receiver reads corrupt/torn data.
- **Mitigation — whether one is needed is FORCED by the caller's data layout ⇒ KNOB, not a fork:**
  - **M2a pre-mcast R→S handshake** — sender `wait`s for receivers' "I consumed it" signal (A6/A7
    on a counter the receivers `up` via A8) *before* mcasting. Required when the dest L1 is a
    **reused** buffer across iters.
  - **M2b none** — omit the handshake when each receiver `reserve`s a **fresh** CB slot per iter
    (the CB protocol already guarantees the dest is free). Required-absent for single-use dests.
- **KNOB `pre_handshake` (use-case):** the caller knows whether its dest is reused. Settle on
  paper (merge-bias), do not bake off. *(This is the `send()` vs `send_into_fresh_slot()` axis.)*

### H3 — Stale flag / exact-match re-trigger (receiver's wait returns on last round's value)
- **Race:** `wait(VALID)` (A6) is exact-match. If the receiver's cell still holds `VALID` from the
  previous round, the next `wait(VALID)` returns immediately — receiver consumes before the new
  data lands.
- **Mitigations (≥2 ⇒ FORK):**
  - **M3a level flag + reset** — receiver resets cell to `INVALID` after consuming each round;
    sender `set_multicast`s `VALID` (A5). Two sem values, one reset write per receiver per round.
  - **M3b monotone counter + `wait_min`** — sender advances a never-reset counter (A9
    `inc_multicast`, or A5 `set_multicast` of an incremented value); receiver `wait_min(round+1)`
    (A7). No reset write; immune to stale-value re-trigger by construction.
- **FORK F2 = flag(+reset) vs counter(+wait_min).** Interacts with the S→R mechanism (H6).

### H4 — Flag-before-data reorder (receiver sees "ready" before data arrives)
- **Race:** if the S→R "data ready" flag could overtake the data mcast, the receiver consumes
  garbage.
- **Mitigation — single, structural ⇒ INVARIANT, not a fork:**
  - **INV4 same-NoC + same-VC FIFO** — A1 data mcast and A5 flag mcast both ride **VC 4** on the
    same NoC ⇒ FIFO arrival, data before flag, at every receiver. The substrate enforces VC 4 for
    both (Step A: no VC override on the mcast path), so the helper gets this **for free** as long
    as it issues data-then-flag on the same `Noc` and does **not** interpose a barrier-less unicast
    path. **The helper must preserve this ordering (issue order = data, then flag).**
- **Interaction with F1:** because INV4 holds, the flag itself acts as the "data landed" proof to
  the receiver — so the *sender's* post-send wait (F1) is only about **source-clobber (H1)**, not
  about the receiver. This is exactly why M1a (flush) is expected to suffice and M1b (barrier) is
  expected to be overkill — the bake-off measures whether that expectation holds across topologies.

### H5 — Degenerate rectangle (self-only / `num_dests == 1`)
- **Race:** loopback variants (A1/A5 INCLUDE_SRC) with `num_dests == 1` are **unspecified — may
  hang** (raw docstring). EXCLUDE_SRC requires ≥1 *other* core in the rect.
- **Mitigation — single safety guard ⇒ always-on, not a fork:**
  - **INV5 degenerate guard** — `if constexpr`/runtime guard: when the rect reduces to self only,
    skip the mcast and do a local copy / `set()` instead. The helper owns this so callers can't
    trip it. *(A coverage trap the bake-off must include as a cell, not a style choice.)*

### H6 — S→R signal mechanism (flag-set vs counter-inc) — coupled to H3
- **Race:** none new; this is *how* the "data ready" advance is delivered, which determines which
  H3 mitigation is even available.
- **Mitigations (≥2 ⇒ FORK, interacts with F2):**
  - **M6a `set_multicast` of a flag value** (A5) — sets each receiver's cell to a level (VALID).
    Pairs with M3a. Source = local sem cell (H1 applies to the cell).
  - **M6b `inc_multicast` of a counter** (A9) — atomically bumps each receiver's cell. Pairs with
    M3b. Sender cannot be in dests (A9) ⇒ interacts with the loopback fork F3.
- **FORK F2′ = set_multicast(flag) vs inc_multicast(counter).** Treat F2 (H3) and F2′ (H6) as **one
  combined fork** — flag-style ⇒ {set_multicast + wait + reset}, counter-style ⇒ {inc_multicast +
  wait_min, no reset}. Bake off the two *styles*, not the four primitive combinations.

### H7 — Sender inside the receiver rectangle (loopback / skip-self)
- **Race / correctness:** when the sender core is geometrically inside the mcast rect:
  - EXCLUDE_SRC (A1) **skips self** — sender's own dest L1 is NOT filled. Wrong if the sender also
    needs the block in its dest buffer.
  - INCLUDE_SRC (A1 loopback) **fills self** too — but `num_dests` counts self and `num_dests==1`
    may hang (H5).
- **Mitigations (≥2 ⇒ FORK):**
  - **M7a EXCLUDE_SRC + sender fills own dest locally** (or sender's dest == its src, no fill
    needed). `num_dests` excludes self.
  - **M7b INCLUDE_SRC loopback** — hardware fills self. `num_dests` counts self.
- **FORK F3 = EXCLUDE_SRC(+local self-fill) vs INCLUDE_SRC(loopback).** The "ANY rectangle"
  generality requires the helper to handle sender-in-rect; *which* way is the empirical question.
  The recognizable predicate (sender ∈ rect? src==dst?) is `constexpr`/runtime-known at the call
  site → a candidate **internal dual-path** if the bake-off shows a tradeoff.

### H8 — num_dests miscount across EXCLUDE/INCLUDE
- **Race:** off-by-one — counting self under EXCLUDE_SRC, or not counting it under INCLUDE_SRC →
  the sender's ACK/flag accounting is wrong (over/under-waits). Does not hang the mcast itself but
  desyncs the handshake counter.
- **Mitigation — single ⇒ INVARIANT:** the helper **derives `num_dests` from (rect, McastMode)**
  in one place, so a caller can never miscount. Tied to F3 (the chosen loopback mode sets the rule).

### H9 — Multiple senders to one receiver (VC ordering is per-sender only)
- **Race:** INV4's same-VC FIFO is **per-sender**. If two senders mcast to the same receiver, their
  data/flag streams interleave arbitrarily — the flag of sender A can arrive before the data of
  sender B.
- **Mitigation — single ⇒ INVARIANT/PRECONDITION:**
  - **INV9 single-sender-per-receiver precondition** — the `Pipe` is a *one-sender* channel to its
    rect. Multi-producer fan-in is out of scope (a documented precondition, not a knob). If a
    caller needs it, they compose multiple `Pipe`s with their own external barrier.

---

## Fork ⇒ bake-off backlog (the ONLY input to Step E.1)

| Fork | Hazard | Variant A (cheap/expected) | Variant B (safe/alt) | Recognizable predicate? |
|---|---|---|---|---|
| **F1 flush vs barrier** | H1 (×H4) | M1a `async_writes_flushed` (SENT) | M1b `async_write_barrier` (ACKed) | global / topology — measure |
| **F2 flag vs counter** | H3 + H6 | M3a/M6a level flag `set_multicast`+`wait`+reset | M3b/M6b counter `inc_multicast`+`wait_min` | per-op staging style |
| **F3 loopback** | H7 (+H8) | M7a EXCLUDE_SRC + local self-fill | M7b INCLUDE_SRC loopback | `constexpr`: sender ∈ rect? src==dst? — dual-path candidate |

## Non-forks (settle on paper / always-on)
- **KNOB `pre_handshake`** (H2) — use-case: dest reused vs fresh CB slot. Merge-bias in Step E.1, no device.
- **INV4** same-NoC+VC-4 data-then-flag ordering — helper preserves issue order.
- **INV5** degenerate-rect guard — always on; appears as a *coverage cell*, not a style.
- **INV8** num_dests derived from (rect, McastMode) in one place.
- **INV9** single-sender-per-receiver precondition — multi-producer out of scope.

## Speculative (bake last / maybe never)
- `linked` on/off — turns out NOT correctness-neutral (see H10). Promoted out of speculative.

---

# AMENDMENTS (Step D consolidation — applied centrally from 6 subagent reports)

The census proved the forks are **richer than the paper catalog assumed**. None of F1/F2/F3 is
speculative — every variant is observed in production. Two new hazards surfaced.

## Amended forks (ternary, not binary)

### F1 — post-send fence: **flush | write-barrier | atomic-barrier** (was binary)
- A **third** value appeared: `noc_async_atomic_barrier` / `Noc::async_atomic_barrier` is the
  correct fence after an **atomic inc/up** (sem counter path), distinct from `async_write_barrier`
  (data path). Observed in sort, argmax, topk, ln_pre_allgather, sdpa read_k, deepseek.
- All three observed both ways across groups (flush in matmul/conv/chain-link; write-barrier in
  group_attn/read_k/rms; atomic-barrier on every counter path). **F1 stays a bake-off fork**, now
  3-valued, and the choice is **coupled to the handshake mechanism** (data→flush/write-barrier,
  counter→atomic-barrier).

### F2 — staging style: **level-flag(+reset) | bounded-counter(wait(N)/set(0)) | monotone-counter(wait_min, no reset)** + **value-carrying payload**
- Confirmed ternary: layernorm sharded uses flag in phase-1 and **monotone counter** (`set(b+2)`/
  `wait_min(b+2)`) in phase-2; argmax runs a monotone `start` counter AND a reset `done` counter
  *simultaneously*; deepseek_prefill uses `inc_multicast`+`wait_min`.
- **Value-carrying flag** (moe_gpt: per-expert token counts packed into the sem word) ⇒ the flag
  payload is arbitrary 32-bit data, not a fixed VALID constant. `receive()` must expose the value.
- F2 is **per-semaphore, not per-pipe** — one block can run a counter on the R→S sem and a flag on
  the S→R sem. The bake-off compares the two *styles*; the API must let a Pipe carry one of each.

### F3 — loopback: **EXCLUDE_SRC | INCLUDE_SRC | degenerate-unicast** (tri-path mandatory)
- Confirmed all three are load-bearing and appear *within single kernels* (matmul block-sharded,
  conv 2d-act, argmax 2-rect with a different mode per rect). The **degenerate case**
  (`num_dests==1`/self-only) is not optional — INV5's guard must collapse to unicast/local-copy or
  the kernel hangs (matmul block-sharded explicit comment). F3 is a strong **internal dual/tri-path**
  candidate; the predicate (sender ∈ rect? rect size==1?) is `constexpr`/runtime-known.

## New hazards

### H10 — `linked`/companion vs barrier mutual-exclusion (NEW, from sdpa)
- **Race/deadlock:** when the data mcast and the sem mcast are issued as a **linked** pair
  (`linked=true`, chain_link.hpp / deepseek mcast.hpp), an intervening barrier **deadlocks** (the
  linked write holds the VC reservation the barrier waits on). When `linked=false` (sdpa read_k), a
  barrier **must** separate data and sem or the flag can overtake the data.
- **Mitigation — two mutually-exclusive contracts (FORK F4, coupled to F1):**
  - **M10a linked pair + flush** — data;sem back-to-back, no barrier between; flush after.
  - **M10b unlinked + barrier-between** — data; barrier; sem.
- **FORK F4 = linked-pair vs unlinked-barrier-between.** Recognizable at the call site; dual-path
  candidate. This is why `linked` is no longer "correctness-neutral."

### H11 — cross-kernel flag reset ownership (NEW, from conv/normalization/data-movement)
- **Race:** the flag is set by the sender but reset by the receiver — and the *timing* varies:
  clear-before-ack (conv2d, ln, gn_v2) vs clear-after-wait (welford, conv3d). Clear-after-wait
  carries a **first-iteration cross-op stale-VALID hazard** (a leftover VALID from a previous op
  on the same L1 triggers an immediate false wait).
- **Mitigation — single ⇒ INVARIANT (helper standardizes):** the two-sided `Pipe` contract **pins
  reset ownership and ordering** — receiver clears **before** signalling readiness
  (clear-before-ack), so the next round always starts from a known-off flag. Not a fork; a
  correctness rule the helper enforces.

## New non-fork structural requirements the API must absorb (not style choices)
These are **generality requirements** surfaced by the census — they go to Step ★ (API feasibility),
not the bake-off:
- **R1 multi-rectangle dest** — send to a *list* of 1–3 rectangles, each with its own McastMode, one
  trailing fence (gn, move, argmax, deepseek).
- **R2 flag-only send** — sem mcast with no data payload (entire data_movement group; ln_pre; gn "go").
- **R3 phase-granular calls** — mcast+handshake is interleaved with DRAM reads/gather; `send()` can't
  be one monolith — the phases (pre-handshake / data / flag / fence) must be separately invokable
  (normalization HOLE).
- **R4 source polymorphism (KEPT) + streaming chunked send (DEFERRED, user).** Source can be a CB
  object, a raw L1 addr (`CoreLocalMem`), or self-CB — KEPT (free: `send`/`mcast_data` take plain
  addresses). A fully-ready block > `NOC_MAX_BURST_SIZE` is auto-chunked by the object API — KEPT.
  **Streaming** a not-yet-complete block (interleave `wait_front` with per-burst mcasts, conv
  `mcast_block_chunked`, to overlap producer with NoC) is **DEFERRED this round** → stays on raw API.
- **R5 NOC1 coordinate swap** — on NOC1 the rectangle coords must be swapped
  (`get_safe_multicast_noc_addr`); ≥5 deepseek kernels — the Pipe must own this.
- **R6 rotating-sender / role-flip** — matmul block-sharded & group_attn: the *same core* is sender
  for block b and receiver for block b′; sender identity rotates per-iteration; `receive()` must take
  a **per-call sender coordinate**, not a fixed one. **This is the sharpest threat to a fixed
  sender-object/receiver-object model** — Step ★ must decide whether the Pipe handles it or defers it.

## Updated bake-off backlog (input to Step E.1)
| Fork | Values | Decided by | Predicate |
|---|---|---|---|
| **F1 fence** | flush / write-barrier / atomic-barrier | coverage+perf; coupled to handshake mechanism | data vs counter path (constexpr) |
| **F2 staging** | level-flag+reset / bounded-counter / monotone-counter (+value payload) | coverage (stale-retrigger) + perf | per-op staging need (knob-ish) |
| **F3 loopback** | EXCLUDE_SRC / INCLUDE_SRC / degenerate-unicast | coverage (degenerate hangs) + perf | sender∈rect?, rect==1? (constexpr) — tri-path candidate |
| **F4 linked** | linked-pair+flush / unlinked+barrier-between | coverage (deadlock vs reorder) + perf | linked (constexpr) — dual-path candidate |

KNOB `pre_handshake` (H2) — confirmed observed BOTH present (dest reused) and absent (fresh CB slot);
settled as a use-case knob, not baked off.
