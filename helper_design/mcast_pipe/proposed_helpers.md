# Step F — Helper Proposal: `Pipe` (`mcast_pipe`)

> **⚠ SUPERSEDED — see `changelog.md` for the authoritative current API.** This Step-F proposal
> exposes `MCAST` (EXCLUDE/INCLUDE) as a caller knob, a single `num_dests`, and ONE `Pipe` object.
> Evolution since: Round 2 dropped `MCAST` and inferred the mode from `num_active_cores`; Round 3
> moved loopback inference to `sender_in_rect && src!=dst`; **Round 4 (2026-06-13)** split the object
> into `SenderPipe` / `ReceiverPipe`, made the count the FULL recipient set (`num_active_receiver_cores`,
> incl. sender-if-receiver), moved semaphore construction+init into the ctors (pass IDs), and put the
> data mcast + self-copy on the `Noc` object (noc 2.0). Authoritative running record: `changelog.md`.

The deliverable. One fat, two-sided helper — `Pipe` — that wraps the NoC-multicast +
semaphore-handshake block, built on the object API (`Noc` / `Semaphore<>` / `MulticastEndpoint`),
with all style choices **decided by the on-device bake-off** (`style_bakeoff.md`), not by argument.

---

## The helper

```cpp
// USE-CASE knob (the op's protocol forces it — NOT a style choice):
enum class Staging {
    Flag,     // level flag (VALID/INVALID, exact wait + reset). DEFAULT — fastest (bake-off F2).
    Counter,  // monotone counter (inc_multicast + wait_min, no reset). For streaming/multi-round
              // protocols that genuinely need it (e.g. layernorm phase-2, census C3).
};

template <
    Staging STAGING       = Staging::Flag,   // use-case knob (default Flag)
    bool    PRE_HANDSHAKE = true>            // use-case knob: dest reused (true) vs fresh CB slot (false)
class Pipe {
public:
    // Topology + handshake semaphores fixed once. `dests` is a list of 1..K rectangles, each with
    // its own McastMode (R1 multi-rect). The Pipe internally: derives num_dests from (rect, mode)
    // [INV8], owns the NOC1 coordinate swap [R5], and collapses a degenerate (num_dests==1 / self-
    // only) rect to unicast/local-copy [INV5 / F3 degenerate].
    Pipe(const Noc& noc, const McastDestSet& dests,
         Semaphore<> data_ready,            // S->R "data is ready"
         Semaphore<> consumed);             // R->S "dest drained" (used iff PRE_HANDSHAKE)

    // ===== DATA channel (a block + a ready-flag) =====
    // send() is atomic and absorbs ALL FOUR guards — callers cannot reorder or skip them:
    //   [if PRE_HANDSHAKE: wait(consumed)]  -> gate the mcast on receivers having drained the dest (H2)
    //   mcast data   (auto-chunks a ready block > burst; STREAMING incremental producer = DEFERRED)
    //   raise flag   (data-before-flag, same VC = INV4; reset ownership = H11)
    //   fence        (flush, F1)
    void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size);
    void receive();   // wait data-ready, reset, [signal slot-free if PRE_HANDSHAKE]. Data is in
                      // the receiver's dst L1 on return; what the caller does with it is its own business.

    // ===== CONTROL channel (a pure flag, no data block) =====
    void     send_signal(uint32_t value = VALID);  // broadcast a control flag (R2)
    uint32_t receive_signal();                     // wait the control flag, RETURN its value:
                                                   // VALID for a plain doorbell; the carried payload
                                                   // for a value-carrying flag (moe_gpt token count).
};
```

> **Symmetric verbs; no sub-methods, no `receive_value`/`receive_drain`.**
> - Earlier drafts exposed `await_consumers` / `mcast_data` / `raise_flag` / `fence` — dropped: the
>   latter three are thin wrappers re-exposing the baked-in decisions (INV4 ordering, F1 fence, H11
>   reset), and **no census kernel interleaves work between them**. `await_consumers` is unnecessary
>   because the consumer-drain wait only needs to precede the **mcast** (the sole writer of the
>   receivers' destination), not the caller's fill — the fill touches the sender's *own* source,
>   protected by the prior send's flush (a separate hazard, H1). So `PRE_HANDSHAKE` doing the wait at
>   the start of `send()` is correct, including for matmul in0.
> - `receive_value` / `receive_drain` were dropped too. **"How does a receiver know whether data
>   accompanied the flag?" → by which verb it calls**, mirroring the sender: `receive()` (data) vs
>   `receive_signal()` (pure flag). The value-carrying flag (moe_gpt) is a *signal with a payload*, so
>   it lives in `receive_signal()`'s return value — keeping the bulk-data `receive()` dead simple
>   (`void`, exact-VALID wait). `receive_drain` ("ack-only, no CB consume") is just a `receive` with no
>   downstream CB work, which the Pipe never owned anyway.
> **(Build-helper to confirm the wait-inside-`send`/`receive` ordering against the real matmul in0.)**

### Use-case sentence (no primitive names)
> *"Broadcast a block from one core to a rectangle of receiver cores and hand it off safely, with
> the sender's `send()` and each receiver's `receive()` two faces of the same channel."*

---

## Baked-in style choices (single path, each cites the bake-off)

| Choice | Decision | Evidence (`style_bakeoff.md`) |
|---|---|---|
| **F1 fence** | `async_writes_flushed()` (SENT), NOT barrier | flush 5576 ns vs barrier 7616 ns = **−27%**, full coverage. Receiver's flag-wait + same-VC FIFO (INV4) already proves arrival; barrier is pure overhead. |
| **F2 staging default** | level **flag** | flag 5505 ns vs counter 7719 ns = **−29%**; counter also needs an atomic barrier + ACK accounting. Counter exposed only as the `Staging::Counter` use-case knob. |
| **flag reset ownership** | receiver clears **before** signalling ready (clear-before-next) | H11 — avoids first-iteration cross-op stale-VALID; the Pipe pins it so callers can't get it wrong. |
| **data→flag ordering** | issue data, then flag, same `Noc`/VC-4 | INV4 — the property that lets the flag prove arrival (so flush suffices). |

## Internal dual-paths (style dispatch on a constexpr predicate — NOT caller knobs; within the ~2 cap)

```cpp
// DUAL-PATH 1 — F4 linking (user-decided "linked where possible" + bake-off confirmed):
//   predicate = can_link  [constexpr: true unless a barrier is structurally required between data
//               and flag — e.g. an unlinked op that fences between them]
//   linked path: data(linked=true); flag(linked=false); flush      -> 3585 ns (fastest, −36%)
//   unlinked fallback: data; barrier-between; flag; flush
if constexpr (can_link) { /* linked pair + flush */ } else { /* unlinked + barrier-between */ }

// DUAL/TRI-PATH 2 — F3 loopback (constexpr predicate sender∈rect / rect size / src==dst):
//   sender ∉ rect (or src==dst)          -> EXCLUDE_SRC               (forced; proven: all coverage)
//   sender ∈ rect, needs own copy        -> INCLUDE_SRC loopback      (MEASURED winner: +26%@32KB,
//                                           +41%@128KB vs EXCLUDE_SRC+local self-copy; tie @1 tile.
//                                           The EXCLUDE+self-copy alternative is empirically rejected.)
//   degenerate (num_dests==1 / self-only)-> unicast / local-copy guard (hang otherwise: docstring + R6)
```

## Forced mechanism (not a fork)
- `Staging::Counter` ⇒ fence is `async_atomic_barrier()` (the non-posted multicast atomic needs ACK
  draining; a write flush **hangs** — bake-off F2). The Pipe wires this automatically with the knob.

## Use-case knobs (the only caller-facing choices)
- `Staging` (default `Flag`) — `Counter` only for monotone/streaming protocols (census C3).
- `PRE_HANDSHAKE` (default `true`) — `false` when each receiver `reserve`s a fresh CB slot per iter.
- `McastDestSet` — the rectangle(s) + per-rect `McastMode`.

## Preconditions / postconditions
- **Pre:** single sender per receiver (INV9 — multi-producer fan-in out of scope); semaphores created
  on the union of sender+receiver cores; `dst_l1` identical across receivers (shared CB index or a
  sharded-tensor base addr). For `Staging::Flag` multi-round with a reused dest, `PRE_HANDSHAKE=true`.
- **Post:** after `send()` returns, the source L1 is safe to reuse (SENT); every receiver's `receive()`
  returns only after its block has landed bit-exact (flag arrival ⇒ data arrival via INV4).

## Fat-helper gate (PASS)
- **≥4 lines replaced:** `send()` absorbs reserve/fill-or-read, data-mcast, flag-stage, flag-mcast,
  fence, plus the F3/F4/degenerate dispatch, NOC1 swap, and num_dests derivation (~10+ call-site lines →
  1). `receive()` absorbs reset, consumed-signal, wait/wait_min, consume (~4–5 → 1). ✔
- **Absorbs ≥1 hazard:** H1 (source-clobber/flush), H3+H11 (stale-flag/reset ownership), H4 (VC order),
  H5 (degenerate guard), H8 (num_dests), H10 (linked/barrier). ✔
- **Nameable without primitives:** yes (use-case sentence above). ✔
- **≤2 helpers:** one `Pipe` object with two faces. ✔

---

## Migration list (from `census.txt` / `migration_audit/_SUMMARY.md`)

**Clean (the spine — migrate first, proves the API):** matmul in0/in1 sender+receiver, conv weights
sender+receiver, ln_post_allgather sender+receiver, topk receiver, sampling, kv_cache, rms_sender,
llama worker_receiver, gn_v2 receiver. `(EXCLUDE or INCLUDE, Flag, flush, pre_handshake known)`.

**Refactor (cost flagged):**
- gn / welford senders — `McastDestSet` multi-rect (R1).
- ln sharded (C3) — `Staging::Counter` + phase-granular calls (R3); two-phase streaming.
- gn interleaved (C4) — double-flag-per-exchange (two `send_signal`/`receive` per iter).
- conv 2d-act / width-sharded — INCLUDE_SRC loopback (F3, handled). **Their streaming chunked
  send (R4) is DEFERRED** — these kernels migrate the loopback/handshake but keep their
  `mcast_block_chunked` (producer-overlapped per-burst broadcast) on raw API this round.
- sdpa read_k — `can_link=false` path (unlinked + barrier-between); deepseek_prefill — `Staging::Counter`.
- move / sort — **legacy raw API → port to `Noc`/`Semaphore` first**, then migrate.

**Defer / out of scope (NOT this round — human-review items):**
- **Rotating-sender / role-flip (R6):** matmul `..._in0_sender_receiver_padding_block_sharded`,
  group_attn — same core sends block b and receives block b′. Needs a **two-Pipe** refactor (one
  sending, one receiving) or stays on raw API. **Confirmed hard by the bake-off** (same-core
  sender+receiver hangs). **Biggest refactor-cost item — flag for the human.**
- Ring/unicast (matmul in0_ring, sdpa ring legs, sort cross-core) — not rectangle-mcast.
- Fabric / cross-chip CCL legs — intent exclusion.
- Preprogram-state perf optimization (deepseek mcast.hpp) — no mcast set-state in object API; future.
- **Streaming chunked send (R4)** — broadcasting a not-yet-complete block by interleaving
  `wait_front` with per-burst mcasts (conv `mcast_block_chunked`), to overlap producer with NoC.
  **Deferred this round** (user decision). The Pipe handles only fully-ready blocks (object API
  auto-chunks a ready block > burst transparently). Source polymorphism (CB / raw-L1 / self-CB) is
  retained — it's free, since `send`/`send_signal` take plain L1 addresses.

## Hand-off
Sign-off on this file **is Gate 1 of `build-helper`**. The bake-off kernels at
`tests/ttnn/unit_tests/kernel_lib/kernels/bakeoff_mcast_{sender,receiver}.cpp` + harness
`test_mcast_pipe.py` carry over as the **raw baseline**; `style_bakeoff.md` carries the coverage/perf
evidence. Provisional (micro-bench) items to confirm against a real op in build-helper: the F1/F2/F4
perf gaps (all cleared ≥10%, but in-context cost may be hidden behind other work — collapse F4 to
safe-global only if the in-context gap falls below threshold). The F3 tri-path and all coverage/hang
verdicts are final (correctness, not provisional).
