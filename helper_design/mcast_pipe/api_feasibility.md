# Step ★ — API Feasibility (`mcast_pipe`)

Intent gave a vague sketch. With the annotated census in hand, this is where the concrete API
takes shape and gets judged: **can a 1–2 helper `Pipe` be built over the `clean`+`refactor` set
without becoming a config-blob?** (Outliers tagged `defer/raw` are out of scope and do NOT count
against feasibility.)

---

## Round-8 re-entry addendum (2026-06-20) — consumer-sem optionality + arg reorder (contract-only)

> DERIVED FROM: feedback-3.txt items 1–4. Re-entry at **Step D** (leftmost — item 2 is a signature /
> param-order / optionality change). Items 1, 3, 4 are **Step G** (implementation: add assert,
> ctor-precompute `sender_in_rect`, hoist the flush) and touch no contract. Authoritative API record =
> `changelog.md` (Round 8). This addendum revises ONE contract point; no pattern coverage changes.

**Claim (item 2):** when `PRE_HANDSHAKE=false`, `CONSUMER_READY_SEM_ID` is constructed but never waited
on (`SenderPipe::send` only `wait`s it under `if constexpr (PRE_HANDSHAKE)`; `ReceiverPipe::receive`
only `up`s it under the same guard). Two **in-scope** call sites prove the dead arg: ln-sharded
`phase1_pipe` (`reader_mcast_sender_unary_sharded_ln.cpp`) and conv-WS `act_mcast_pipe`
(`activation_reader_width_sharded.cpp`) both pass `PRE_HANDSHAKE=false` and are forced to supply a
consumer sem the Pipe ignores. (Invariant 5 satisfied — the `false` arm has real users.)

**Decision (user pick — keep the named knob, make the sem optional, push the rarest knob last):**
`CONSUMER_READY_SEM_ID` becomes a **trailing param defaulted to an `UNUSED_SEM_ID` sentinel**, guarded
by `static_assert(!PRE_HANDSHAKE || CONSUMER_READY_SEM_ID != UNUSED_SEM_ID)`. `PRE_HANDSHAKE` stays an
explicit bool and moves **before** the sem (gate-then-resource). `DATA_READY_SIGNAL` moves to **last**
(its `Counter` arm is the rarest, most-defaulted knob).

**Revised contract — the only change to the draft:**

```cpp
// UNUSED_SEM_ID = a reserved sentinel id (no real CTA sem uses it) meaning "no consumer sem".
template <
    uint8_t  NOC_ID,                                       // required
    uint32_t DATA_READY_SEM_ID,                            // required
    uint32_t NUM_ACTIVE_RECEIVER_CORES,                    // required
    bool     PRE_HANDSHAKE       = true,                   // gate
    uint32_t CONSUMER_READY_SEM_ID = UNUSED_SEM_ID,        // required IFF PRE_HANDSHAKE (static_assert)
    DataReadySignal DATA_READY_SIGNAL = DataReadySignal::Flag>   // rarest knob, last
class SenderPipe;   // static_assert(!PRE_HANDSHAKE || CONSUMER_READY_SEM_ID != UNUSED_SEM_ID, ...)

template <
    uint32_t DATA_READY_SEM_ID,                            // required
    bool     PRE_HANDSHAKE       = true,                   // gate
    uint32_t CONSUMER_READY_SEM_ID = UNUSED_SEM_ID,        // required IFF PRE_HANDSHAKE (static_assert)
    DataReadySignal DATA_READY_SIGNAL = DataReadySignal::Flag>   // rarest knob, last
class ReceiverPipe;   // same static_assert
```

Call-site forms after the reorder:
- handshake sender (matmul/conv-weights): `SenderPipe<NOC, DR, NUM, true, CR>`
- no-handshake sender (ln phase1 / conv-WS): `SenderPipe<NOC, DR, NUM, false>`   ← drops the dead sem
- counter sender: `SenderPipe<NOC, DR, NUM, true, CR, DataReadySignal::Counter>`
- handshake receiver: `ReceiverPipe<DR, true, CR>`   ·   no-handshake receiver: `ReceiverPipe<DR, false>`

Side effect (improvement): the all-default form `SenderPipe<NOC, DR, NUM>` now **fails the
static_assert** (`PRE_HANDSHAKE=true` + no sem) — a control-only sender like topk (`send_signal`, never
gates) must declare `PRE_HANDSHAKE=false` honestly, which is more correct than its current default-true.

**Verdict: FEASIBLE WITH REVISION (contract-only).** No pattern coverage changes; no style fork touched.
Caller-facing (reordered + optional arg) → `MCAST_PIPE_API_VERSION` bump (6→7). Step E is a re-confirm
no-op (no device). Items 1/3/4 are internal-only (no contract change, no bump on their own).

---

## Round-6 re-entry addendum (2026-06-20) — flag-set lifecycle + arg order (contract-only)

> DERIVED FROM: feedback.txt items 1, 2, 5. Re-entry at **Step D** (leftmost — signature + param
> order + count/flag semantics). Items 3/4/6 are Step-F wording. The authoritative API is
> `changelog.md` (Round 6). This addendum revises the contract; no pattern coverage changes.

**Claims (items 1, 2, 5):**
1. The per-send local `data_ready.set()` is needed only off the loopback path, and there only ONCE.
2. `INITIAL_FLAG_VALUE` is dead (the per-send `set` always overwrote the ctor init) — drop it; keep a
   ctor `set(VALID)` for the no-loopback case.
5. Reorder the SenderPipe template args.

**Why it's a contract change, confirmed in code:** `Semaphore<>::set_multicast` (`noc_semaphore.h:165`)
broadcasts the sender's **local cell** as its source — no `value` arg. For the Flag signal the source
is always `VALID`, so it is set ONCE in the ctor and reused (the proven raw matmul does exactly this:
`reader_bmm_tile_layout_in0_sender_padding.cpp:53` sets `VALID` once before the loop). `INITIAL_FLAG_VALUE`
could never reach the wire → drop the param.

**Revised contract (the only change to the draft):**
- Drop `INITIAL_FLAG_VALUE`. Ctor sets the sender's local data-ready cell `= VALID` once (Flag only).
  `send()` does no per-send local set; `send_signal()` is a plain doorbell (no `value` param).
- SenderPipe template arg order: `NOC_ID` (no default, first) → `DATA_READY_SEM_ID` →
  `CONSUMER_READY_SEM_ID` → `NUM_ACTIVE_RECEIVER_CORES` → `DATA_READY_SIGNAL` (=Flag) → `PRE_HANDSHAKE`.

**Verdict: FEASIBLE WITH REVISION (contract-only).** No pattern coverage changes; no style fork is
touched. Caller-facing (removed param + reordered args + `send_signal` signature) → `MCAST_PIPE_API_VERSION`
bump (5→6). Step E is a re-confirm no-op; no device work.

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

---

# Step ★ — Round-7 addendum: TOPOLOGY SURVEY + CAPABILITY MATRIX (feedback-2.txt)

> DERIVED FROM: the full `census.txt` + `kernel_annotations/*` (topology lens) · `primitive_contracts.md`
> A5 vs A5′ (set_multicast vs relay_multicast) · `hazards_catalog.md` H12/INV12 · current API = v6.
>
> **Why this addendum exists.** Earlier rounds resolved the *style* forks (F1/F2/F3/F4) but never
> enumerated the **topologies** the abstraction must support, so the CHAIN cross-id-relay gap was
> captured only *implicitly* (folded into the F2=FLAG tag of `chain_link.hpp.md`) and never surfaced
> as a first-class capability gap. This survey makes every topology's verdict **explicit** so missing
> capabilities are recorded once, not re-discovered per kernel. **No API change this round** — the
> chain family stays deferred; this is the map of what the Pipe covers vs. what it must grow into.

## Part 1 — the distinct topologies (primary axis)

| # | Topology | Mechanism | Sender also a receiver? | Verdict | Where observed |
|---|----------|-----------|--------------------------|---------|----------------|
| T1 | **STAR** — 1 sender → N pure receivers, **shared** sem id | A5 `set_multicast` of the sender's OWN cell (src==dst id) | No | **SUPPORTED** (current `SenderPipe`/`ReceiverPipe`, v6) | matmul, conv, gn, ln, topk, deepseek samplers (13 migrated) |
| T2 | **CHAIN** — store-and-forward, each link **receives AND forwards** | A5′ `relay_multicast`: write-once `valid_sem` (src) → next link's `receiver_sem` (dst), **src≠dst** | **Yes** | **GAP** | chain_link.hpp (ref), reader_interleaved (refactor), exp_ring_joint_reader (refactor) |
| T3 | **RING / all-gather** — chain + co-located per-link ring sync | T2's relay **plus** a `wait_min` ring barrier + mux-writer fabric path | Yes | **GAP** (chain half) **+ OOS** (ring sync — cleanly separable, composed externally) | exp_ring_joint_reader (ring `wait_min` L271/L430) |
| T4 | **FABRIC mcast** — cross-chip broadcast | fabric/CCL data path (not local NoC rectangle mcast) | n/a | **OUT OF SCOPE** (different family — intent exclusion) | all_reduce worker_writer (sem rectangle is local but coupled to fabric) |
| T5 | **Multi-producer fan-in** — N senders → 1 receiver | N independent A1/A5 streams into one cell | n/a | **OUT OF SCOPE** (INV9 precondition — VC-FIFO order is per-sender only) | none in census; documented precondition |

**The single root cause of T2/T3 being a GAP (not T1):** in T1 the sender is *not* a receiver, so its
doorbell cell is free scratch — A5 (same-id `set_multicast`) is correct and simpler. In T2/T3 every
link is *both*, its doorbell is **mutable** (`receive()`→INVALID), so the relay needs a **separate
write-once `valid_sem`** as source (cross-id A5′). The current Pipe has exactly one shared `data_ready`
sem and only does A5 → it **structurally cannot** express T2/T3 (A5′'s `ASSERT(src != dst)`). See
H12/INV12. relay buys **no perf** for T1 — it earns its keep *only* where src/dst ids must differ.

## Part 2 — fine matrix (topology × style sub-axes)

Each cell = does the **current v6 Pipe** cover that combination? `✓`=supported, `GAP`=needs relay/new
capability, `OOS`=out of scope.

| Topology | F1 fence | F2 handshake (flag / counter) | F3 loopback (excl / incl / degenerate) | pre_handshake (yes / no) |
|----------|----------|-------------------------------|------------------------------------------|---------------------------|
| **T1 STAR** | ✓ flush+linked baked (F4); barrier dialects converge to linked-flush — see note | ✓ both (`DataReadySignal=Flag\|Counter`); value-carrying flag → deferred (moe_gpt) | ✓ tri-path internal dispatch (inferred `sender_in_rect && src!=dst`); degenerate→local copy | ✓ both (`PRE_HANDSHAKE` knob) |
| **T2 CHAIN** | **GAP** — chain uses flush+linked, but it rides A5′ relay the Pipe lacks | **GAP** — needs the **two-sem** model (write-once `valid_sem` + mutable `receiver_sem`), not one shared id | **GAP** — chain is EXCLUDE-style (injector self-fills via DRAM), but unreachable without relay | **GAP** — chain *is* pre_handshake=YES (dest reused), but blocked on relay |
| **T3 RING** (chain half) | **GAP** | **GAP** | **GAP** | **GAP** |
| **T3 RING** (ring sync) | **OOS** — `wait_min` ring barrier is NOT a Pipe responsibility; stays cleanly separable from the relay channel | OOS | OOS | OOS |
| **T4 FABRIC** | OOS | OOS | OOS | OOS |
| **T5 fan-in** | OOS | OOS | OOS | OOS |

**Note on T1 F1 fence dialects.** Three fence dialects appear in STAR kernels (flush, write-barrier,
atomic-barrier — `hazards_catalog.md` F1 amendment). The Pipe bakes **flush + always-linked** (Round 1
−27%, Round 4 P5). The one observed BARRIER+unlinked STAR dialect (sdpa_decode `read_k`) was found to
have **no genuine consumer** — it converges to linked-flush and would gain the −36% win (changelog
Round 4 P5). So within T1 these are not separate *supported paths* but a single baked path the
refactor targets converge onto; atomic-barrier remains the counter-path fence. No T1 cell is a GAP.

## Verdict (Round 7)

- **T1 STAR — FULLY SUPPORTED.** Every (F1 × handshake × loopback × pre_handshake) sub-cell in scope
  is covered by the v6 `SenderPipe`/`ReceiverPipe`. No gap.
- **T2 CHAIN — EXPLICIT GAP.** Requires cross-id relay (A5′ `relay_multicast`) + a **two-sem per link**
  model (write-once `valid_sem` source ≠ mutable `receiver_sem` doorbell). The current single-shared-sem
  star Pipe cannot express it. **Chain family stays DEFERRED** (reader_interleaved, exp_ring_joint_reader,
  chain_link.hpp reference). Closing the gap is a future capability round — likely a third face
  (a `RelayPipe`/forwarding link) or a relay mode on `SenderPipe`, decided when the chain family is
  actually scheduled for migration.
- **T3 RING — GAP + OOS.** Chain half = T2's gap; the co-located ring `wait_min` sync is **out of scope**
  and must remain externally composed, never absorbed into the Pipe.
- **T4 FABRIC / T5 fan-in — OUT OF SCOPE**, now explicitly bounded rather than silently ignored.

**No bake-off, no API change, no version bump this round** (E and G are no-ops): relay is a
topology-forced capability, not a style fork measurable on the star, and the chain family is deferred.

---

## Round-9 addendum (2026-06-20, feedback-4.txt) — NO API change

DERIVED FROM: hazards_catalog H12 amendment (M12b) · proposed_helpers Round 9 · changelog Round 9.

Re-entry routed at **Step B** (H12's mitigation set was incomplete for the rotating-role STAR). The
fix — re-assert the sender's own `data_ready` cell = VALID per send on the **Flag path** before the
flag `set_multicast` — is **internal to `SenderPipe::send()`/`signal_ready_`**. It touches **no**
caller-facing surface:

- **No template-arg change** — same `SenderPipe`/`ReceiverPipe` signatures (version 7).
- **No new knob, no new face** — explicitly NOT gated behind a predicate (M12b is DOMINANT: redundant
  no-op store for the pure STAR, required for the rotating STAR). A two-Pipe rotating-role call site
  uses the *existing* `SenderPipe` + `ReceiverPipe` unchanged.
- **No count-semantics change** — `NUM_ACTIVE_RECEIVER_CORES`, loopback derivation, ack accounting all
  unchanged.

**Feasibility verdict — unchanged: T1 STAR FULLY SUPPORTED**, now *including* the rotating-role STAR
sub-case (two Pipes on a shared cell), which the Round-7 survey had implicitly conflated with the
CHAIN gap. CHAIN (T2) is still a GAP — it needs INV12 cross-id relay because its source ≠ dest cell;
the star can re-store its own cell, so M12b suffices. **No bake-off cell (re-decide only), no API
change, no `MCAST_PIPE_API_VERSION` bump** (G is an internal `send()`-body edit).
