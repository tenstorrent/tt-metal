# Step ★ — API Feasibility (`mcast_pipe`)

Intent gave a vague sketch. With the annotated census in hand, this is where the concrete API
takes shape and gets judged: **can a 1–2 helper `Pipe` be built over the `clean`+`refactor` set
without becoming a config-blob?** (Outliers tagged `defer/raw` are out of scope and do NOT count
against feasibility.)

---

## Round-5 re-entry addendum (2026-06-19) — `McastRect` is templated on the NoC id

> DERIVED FROM: feedback.txt item 1. Re-entry at **Step D** (contract change, leftmost). The rest
> of this file (the old `Pipe` + `McastDestSet` draft below) is the historical Step-D record;
> the authoritative API is `changelog.md`. This addendum revises ONE contract point.

**Claim (item 1):** `McastRect::start_end_for_noc(noc_id)` runs a corner comparison + per-NoC swap
on *every* `send()` (twice — once for the data mcast, once for the flag mcast). The NoC id is
compile-time (`constexpr uint8_t noc_index = NOC_INDEX`, `dataflow_api_common.h`) and the factory
chooses it, so the comparison can be hoisted out of the hot path entirely.

**Revised contract — `McastRect<uint8_t NOC_ID = noc_index>`:**
- Adds a compile-time template param `NOC_ID`, defaulted to the kernel's `noc_index` so the common
  call site writes `McastRect<>{x0,y0,x1,y1}`; a factory that drives a specific NoC passes it
  explicitly (`McastRect<1>{...}`). The four coords stay **runtime** (per-core — invariant 6 honoured).
- The **constructor** computes and stores, ONCE, both the routing-correct `(start,end)` for `NOC_ID`
  *and* the normalized box (`xlo..yhi` for the containment test). The per-call comparison/swap is gone.
- The runtime method `start_end_for_noc(noc_id)` is **deleted**; callers/`SenderPipe` read stored
  fields. `SenderPipe` gains the same `NOC_ID` template param (defaulted to `noc_index`) so even
  `sender_in_rect_`'s `my_x[noc_.get_noc_id()]` collapses to a compile-time `my_x[NOC_ID]`.

**Verdict: FEASIBLE WITH REVISION (contract-only).** No pattern coverage changes; no style fork is
touched. This is a representation/type-honesty change — the same address math, computed once at
construction instead of per call. Caller-facing (the `McastRect` ctor/type), so it forces a call-site
rewrite → `MCAST_PIPE_API_VERSION` bump (Step G). Step E is a re-confirm no-op (below); no device work.

---

## First concrete API draft (promoted from the Step-0 sketch)

Grounded in the two prior-art shapes the census found (`deepseek_b1_ops::Mcast` — CT-templated
two-sided object with role dispatch; sdpa `ChainLink`). Rebuilt on the **object API** substrate.

```cpp
// ---- compile-time style axes (bake-off winners become the DEFAULTS; some are use-case knobs) ----
enum class Staging  { Flag, Counter };            // F2 use-case: level-flag+reset | monotone wait_min
enum class Fence    { Auto, Flush, WriteBarrier, AtomicBarrier };  // F1: Auto dispatches by path
enum class Linking  { Auto, LinkedPair, UnlinkedBarrier };         // F4: Auto by linked predicate

template <
    Staging  STAGING       = Staging::Flag,   // use-case knob (op's protocol decides)
    bool     PRE_HANDSHAKE = true,            // H2 use-case knob (dest reused vs fresh CB slot)
    Fence    FENCE         = Fence::Auto,     // F1 bake-off default; internal dispatch by path
    Linking  LINK          = Linking::Auto>   // F4 bake-off default; internal dispatch by predicate
class Pipe {
public:
    // Topology + semaphores fixed once. dests is a LIST of 1..K rectangles (R1), each with its
    // own McastMode; the Pipe derives num_dests per rect (INV8) and owns NOC1 coord-swap (R5)
    // and the degenerate-rect→unicast collapse (INV5 / F3 degenerate).
    Pipe(const Noc& noc,
         const McastDestSet& dests,           // {rect, McastMode}[]  — EXCLUDE_SRC|INCLUDE_SRC per rect
         Semaphore<> data_ready,              // S->R  ("data is ready")
         Semaphore<> consumed);               // R->S  ("I drained the dest") — only used if PRE_HANDSHAKE

    // ===== SENDER side =====
    // send() is atomic: [if PRE_HANDSHAKE: wait(consumed)] -> mcast data -> raise flag -> fence.
    // All four guards are internal (no public sub-steps): the wait gates the mcast (H2), the flag
    // follows data on the same VC (INV4), reset ownership is pinned (H11), fence = flush (F1).
    void send(uint32_t src_l1, uint32_t dst_l1, uint32_t size);
    void send_signal(uint32_t value = VALID);                     // R2 flag-only (no data payload)

    // ===== RECEIVER side (symmetric with the sender) =====
    void     receive();          // pairs with send():        wait data-ready, reset, [signal slot-free]
    uint32_t receive_signal();   // pairs with send_signal():  wait control flag, RETURN its value
                                 //   (VALID, or a value-carrying payload e.g. moe_gpt token count)
    // (a per-call-sender `receive_from(...)` was considered for the rotating-sender hybrid, but
    //  R6 is scoped OUT — see verdict — so it is NOT part of the API.)
};
```

Source polymorphism (R4): `src_l1`/`dst_l1` are plain L1 addresses, so a CB write-ptr, a raw
`CoreLocalMem` addr, or self-CB (src==dst) all work without overloads.

---

## Per-pattern coverage map (the ★ gate)

| Pattern (census) | Covered? | How |
|---|---|---|
| **P1/C1 canonical** (matmul in0/in1, conv weights, ln_post) | YES as drafted | `send()`/`receive()`, Flag, EXCLUDE or INCLUDE |
| **C2 bounded-counter + pre-handshake, multi-rect** (gn, welford) | YES w/ revision | `McastDestSet` (R1) + `PRE_HANDSHAKE=true`; counter is a bounded form of `Staging::Counter` |
| **C3 two-phase flag→monotone** (ln sharded) | YES w/ revision | the two phases are **two separate `send()`/`send_signal()` calls** (not cracked-open sub-steps) + `Staging::Counter` (`wait_min`) for phase-2 |
| **C4 double-flag-per-exchange** (gn) | YES w/ revision | a `send_signal`/`receive_signal` "go" pair + a `send`/`receive` data pair per iter |
| **chain_link / read_k** (sdpa) | YES as drafted | `Linking::Auto` dispatches LinkedPair (chain) vs UnlinkedBarrier (read_k) — **F4 internal** |
| **flag-only** (data_movement, ln_pre, sort, argmax) | YES w/ revision | `send_signal()` — data payload optional (R2) |
| **value-carrying flag** (moe_gpt) | YES w/ revision | `send_signal(value)` / `receive_signal()` returns the payload |
| **chunked send > burst, READY block** (conv 2d-act) | YES as drafted | object API auto-chunks a ready block transparently |
| **STREAMING chunked send, incremental producer** (conv `mcast_block_chunked`, R4) | **DEFERRED** (user) | producer-overlapped per-burst broadcast stays on raw API this round |
| **ack-only/no-consume receive** (conv3d) | YES as drafted | just `receive()` / `receive_signal()` with no downstream CB work — the Pipe never owned CB-consume |
| **multi-rect, per-rect mode** (gn, move, argmax, deepseek) | YES w/ revision | `McastDestSet` carries per-rect `McastMode` (R1) |
| **NOC1 coord swap** (deepseek ×5) | YES (internal) | Pipe owns it (R5) — not caller-visible |
| **degenerate self-only** (matmul block-sharded) | YES (internal) | INV5: collapse to unicast/local-copy (F3 degenerate) |
| **ROTATING-SENDER / role-flip** (matmul block-sharded, group_attn) | **NO — scoped out** | a single core that is sender for block b and receiver for block b′ needs **two Pipe roles live at once on one core** → see verdict |
| **preprogram-state mcast** (deepseek mcast.hpp, kv_cache) | Functionally YES; **optimization deferred** | object-API `async_write_multicast` has **no mcast set-state** (Step A) — the persistent-descriptor perf trick is out of object-API scope; plain re-issue is correct, just not the fast path |
| **ring/unicast, fabric CCL** (matmul in0_ring, sdpa ring legs, all_reduce, all_to_all fabric) | OUT | not rectangle-mcast / cross-chip — intent exclusion, tagged defer/oos |

---

## Verdict: **FEASIBLE WITH REVISION**

A 1–2 helper `Pipe` covers the entire `clean`+`refactor` set **provided** the API is revised from
the naive `send(src,dst,size)` sketch to the draft above. The revisions are **use-case knobs**
(`STAGING`, `PRE_HANDSHAKE`, `McastDestSet`), a **symmetric control channel** (`send_signal` /
`receive_signal`), and **internal dispatch** (FENCE, LINKING,
degenerate-unicast, NOC1-swap, ready-block chunking) — **not** a parameter explosion the caller must navigate.
It survives the config-blob test because:
- the four **style forks (F1/F3/F4)** are *internal dispatch on constexpr predicates*, invisible to
  the caller — exactly what the bake-off resolves;
- only **two** caller-facing *style-ish* knobs remain (`STAGING`, `PRE_HANDSHAKE`), and both are
  genuine **use-case** differences (the op's protocol forces them), not style;
- the rest is overloads/optional-payload, which is normal API surface, not configuration.

### Scoped OUT (does not count against feasibility)
1. **Rotating-sender / role-flip same-core** (matmul `..._in0_sender_receiver...`, group_attn) —
   `refactor`/`defer`. Even with a per-call sender coordinate, a core that is simultaneously
   sender(b) and receiver(b′) within one loop is a **two-live-roles** shape that a single `Pipe`
   object intentionally does not model. These kernels either instantiate **two Pipes**
   (one sending, one receiving) — viable but a real refactor — or stay on raw API this round. **This
   is the one yellow flag**: ~2 high-value matmul kernels need the two-Pipe refactor; flagged for the
   human, not papered over.
2. **Preprogram-state perf optimization** — out of object-API scope (no mcast set-state exists).
   Functionally covered; the fast path is a separate future concern.
3. **Ring/unicast + fabric CCL** — intent exclusion.
4. **Streaming chunked send (R4)** — user-deferred. Broadcasting a not-yet-complete block by
   interleaving `wait_front` with per-burst mcasts to overlap producer with NoC (conv
   `mcast_block_chunked`). The Pipe handles only fully-ready blocks (object API auto-chunks a ready
   block > burst). Source polymorphism (CB / raw-L1 / self-CB) is retained — free via plain addresses.

### Bake-off backlog the API leaves open (→ Step E)
The API draft fixes everything *except* the style-fork **defaults** it dispatches on. Step E decides:
- **F1 FENCE** — flush vs write-barrier vs atomic-barrier (and the `Auto` dispatch rule by path).
- **F3 loopback** — EXCLUDE_SRC vs INCLUDE_SRC vs degenerate-unicast (the tri-path predicate + perf).
- **F4 LINKING** — LinkedPair+flush vs Unlinked+barrier-between (the `Auto` predicate + deadlock/perf).
- **F2 STAGING** — flag vs counter: *coverage* (stale-retrigger immunity) is already a known
  correctness axis; the bake-off measures whether one is also a perf win, else it stays a use-case knob.
