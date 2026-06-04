# Ring Joint SDPA — Chunked-Prefill k-stationary (Q/K loop) Reorder Feasibility

> **RESULT — LANDED 2026-06-04 (the reorder was NOT needed; a better, contained fix landed instead).**
>
> Per §5 step 2 of this doc ("do the V^T-from-K^T compute sourcing change first … independently a larger
> measured win"), I implemented **V^T-from-K^T** instead of the reorder. Outcome on the target config
> (kimi50k-q32-k640-chunk2560, QB 4×p150b, 100 cores):
>
> | | Duration | Math Util | PCC |
> |---|--:|--:|--:|
> | baseline | 3.191 ms | 59.4 % | 0.99947 |
> | **V^T-from-K^T (landed)** | **2.991 ms** | **63.4 %** | **0.99947 (byte-identical)** |
> | Δ | **−0.200 ms (−6.3 %)** | **+4.0 pt** | exact |
>
> This **beats the reorder's own proven ceiling (2.999 ms / 63.2 %)** with a contained, hang-free,
> kernel-only change (no build, no factory/semaphore/CB-layout churn → none of §4's `d33553875fa` hang
> vector). It is a pure addressing remap: in latent mode `cb_v_in == cb_kt_in`, and `materialize_v` is a
> verbatim whole-tile copy, so `V[sk][vd]` is byte-identical to `cb_kt_in[vd*KT_stride + sk]`. The S@V
> matmul now reads V^T directly from the K^T buffer (one output column per matmul since vd is
> KT_stride-strided → ct_dim=1), the Phase-1 K^T pop is deferred to after Phase 2, and the reader stops
> materializing the V entry. Gated behind `chunked_enabled && v_shares_k_buffer`; all other paths untouched.
>
> **The reorder is now retired for this config.** With all DM disabled on the new kernel the compute
> ceiling is **2.950 ms / 64.3 %**, so the remaining exposed data-movement gap is only **0.041 ms / 0.9 pt** —
> the reader is fully hidden behind compute. The k-stationary reorder only reduces *reader* K-traffic, so
> its maximum possible additional gain is 0.041 ms, not worth the 2-to-3-sided hang-prone restructure.
> (Removing V-remat dropped the reader from ~23 µs/chunk to ~18 µs/chunk, below the ~21.65 µs compute step.)
> The one residual: ct_dim=1 raised the compute ceiling itself by ~0.069 ms (2.881→2.950) vs the old blocked
> V matmul; recoverable only with a batched-vd DST matmul if dst_size ≥ vDHt — marginal, not pursued.
>
> Patch: `/tmp/phaseA_vfromkt.patch`. Files: `compute_streaming.hpp`, `ring_joint_reader.cpp`.

Branch: `skrstic/ring_joint_sdpa_chunked_perf_sweeps`
Date: 2026-06-04
Config under study: **sp=4 ring-of-4, latent V, heads/device=14, per-device seq=640, q_chunk=32
(Sq_chunk_t=1), k_chunk=640 (Sk_chunk_t=20), d_q=576 (DHt=18), d_v=512 (vDHt=16), NHK=1**, num_q_chunks=20,
**q_per_core(max)=3**, iter_num_kv_chunks≈11, K mcast enabled.

Companion to `chunked_prefill_dv512_dm_ablation_findings.md`. That doc proved the exposed data-movement penalty
(~0.31 ms / 6.4 pt) is almost entirely the **redundant K traffic**: because NHK=1, the K slice is identical across
the 3 Q chunks a core owns, so the reader DRAM-reads + transposes + mcasts each K chunk **3× redundantly per
ring_iter**. The principled fix is the **k-stationary reorder**: invert the loop nest from `for q → for k_chunk`
(q-outer) to `for k_chunk → for q` (k-outer) so each K chunk is read+mcast **once** and the 3 Q chunks consume it
in place. Proven timing-only ceiling: **≈2.999 ms / 63.2 %** (vs 3.19 ms / 59.4 % baseline; recovers ~62 % of the
gap to the 2.881 ms / 65.8 % compute ceiling).

This document is a deep, four-pronged feasibility audit of that reorder: **(1) reader, (2) compute, (3) L1 budget,
(4) chain/mcast + fused-op coupling.** It is based on a full read of the kernels and program factory, not the
prior doc's summary.

---

## TL;DR

**The reorder is algorithmically clean but is a 2-to-3-sided (reader↔compute↔writer) restructure with a proven
hang vector. It is feasible. It is not a quick edit.**

The four findings, ranked by how much they change the prior understanding:

1. **L1 is NOT the hard blocker (correction to the prior doc).** The prior doc concluded the reorder is "L1-blocked"
   (118 KB free, 3 ping-pong accumulators need ~131 KB → OOM). That is true *only without* the K-CB depth-3→2
   reclaim. **With** the depth reclaim (frees 391,680 B = 382.5 KB): single-buffered per-Q accumulators fit with
   **~465 KB margin**, and even *ping-pong* per-Q accumulators fit with **~357 KB margin**. The L1 numbers in the
   prior doc reproduce exactly from source (1,451,536 B total, 118.5 KB free, cb_k_in = 81 %), but the "boxed in"
   conclusion was drawn before crediting the reclaim against the accumulator cost. The real open question on L1 is
   *behavioral*, not capacity: does dropping the K-CB 3rd slot still let V-remat hide once compute spends 3× longer
   per k_chunk (see §3)?

2. **The hard blocker is the resident-K / pop-cadence contract — it forces a 2-sided change.** Today compute pops
   K (`cb_kt_in`) and V (`cb_v_in`) once per **(q, k_chunk)** step, and `cb_k_in` is sized **3-deep, not
   iter_num_kv_chunks-deep**. So "read+push K once" is impossible without *also* changing compute to consume one
   resident K-CB entry for all 3 q's before popping. The loop swap cannot land reader-only.

3. **The dominant residual risk is fused-op/AllGather positional-layout coupling, not the mcast arithmetic.** The
   mcast rebalance itself is tractable (1-hop row mcast, reset-per-chunk rendezvous, no cross-link accumulation).
   But the SDPA op shares one ordered `desc.semaphores` ID pool, one positional RT-arg stream, and one CB-index
   space with the fused AllGather op. Adding/removing/reordering *any* semaphore, RT arg, or CB shifts every
   downstream offset and silently points a kernel at the wrong address → hang. This is exactly what hung the prior
   K-mcast-to-writer attempt (commit `d33553875fa`).

4. **SALAD is not in-place + the writer handshake is q-interleaved.** The flash-attention online-softmax math
   reorders cleanly (causal geometry is stateless per (k,q)), but `salad_correct_fused` reads `prev.out` and writes
   a *different* `out_cb` — not in place — so per-Q accumulation needs either a 2nd buffer per Q or an in-place
   SALAD rewrite. And because each Q finishes its causal K-range at a *different* outer k_chunk, the
   compute→writer `cb_signal` "last-K, start draining" handshake becomes interleaved across Q's and must be made
   q-aware → pulls the writer into the change (3-sided).

**Verdict:** every algorithmic sub-problem is solvable; the cost is a coordinated reader+compute(+writer) rewrite
behind a tight config gate, validated against PCC, with several expected device-reset/hang cycles. The prior doc's
"L1-boxed, genuinely boxed in" framing **overstated the capacity wall** and **understated the
control-flow/coupling work** — the reorder is gated by engineering effort and hang-risk, not by L1 bytes.

---

## 1. Reader side (`ring_joint_reader.cpp`)

### Current loop nest
- **L359 `for ring_iter`** → per-iter: `ring_id`, `iter_num_kv_chunks` (L393), `ring_iter_kv_start_tile` (L377).
- **L418 `for q_iter` (0..loop_q_count)** — `loop_q_count = max_q_per_core` under K-mcast (L416);
  `is_padded_iter = q_iter >= q_per_core` (L420); per-q: `q_slice`, `nb/nq/q_chunk`, `q_iter_local=q_iter` (L457).
- **L463 `for k_chunk` (0..iter_num_kv_chunks)** — K reserve (L510-517), K receive/fetch (L519-533),
  K forward/mcast (L536-538), `cb_push_back(cb_k_in)` (L575), Q-read at `k_chunk==0` (L582-616),
  V-remat (L618-642). Post-loop `dummy_kv_chunks` drain (L673-684).

### The redundancy is provably q_iter-invariant
With NHK=1, `q_heads_per_k = NH`, so `nk = nq / q_heads_per_k == 0` for all q_iter (L484). The K slice
(L487-489 ring0 / L494-496 ring>0) is `Slice(nb, nk, k_chunk*Sk_chunk_t, …)` — **q_iter does not enter it at
all** (the only q-derived index is `nb`, and B=1 here so `nb==0` always). The chosen `k_gen` depends only on
`kv_chunk_is_joint`/`ring_iter`. ⇒ the fetched K^T data and the mcast payload are **bit-identical across all
q_iter** for fixed (ring_iter, k_chunk). V-remat (`materialize_v_prefix_from_k`, reads only `cb_k_start_address`)
is likewise q-invariant in the chunked path. **The 3× redundancy is real and removable.**

### What must change (and why it isn't reader-only)
Inverting to `for ring_iter → for k_chunk → for q_iter`:
- K receive/fetch/forward (L510-538) → hoist to **once per k_chunk**. *(the intended win)*
- `cb_push_back(cb_k_in)` (L575) → **cannot stay once.** Compute does `cb_wait_front`+`cb_pop_front(cb_kt_in)`
  per (q, k_chunk) (compute L936/L2028), and `cb_k_in` is only **3 entries deep** (factory L356:
  `Sk·DHt·(v_shares_k_buffer?3:2)`), *not* iter_num_kv_chunks-deep. So a single fetched K chunk must stay live
  across the inner q-loop while being consumed 3×. Two options:
  - **(A) 2-sided:** resize `cb_k_in` semantics + change compute to pop K only on the last inner q. *(clean, invasive)*
  - **(B) reader-mostly:** read DRAM once into a private L1 scratch, re-copy into CB staging per q (saves the DRAM
    read+transpose+mcast, costs an L1 buffer + an L1→L1 copy/q). Spends the scarce dv512 L1.
- Q-read (L582-616): currently keyed `k_chunk==0`; must become q-keyed inside the new inner loop, preserving
  compute's q-order into `cb_q_in`.
- V-remat / V-chain (L618-670): V is q-invariant; same resident/re-push problem as K (and `cb_v_in` aliases
  `cb_k_in` in latent-V mode).
- `dummy_kv_chunks` drain (L673-684) and the padded-iter block (L546-572): re-derive (see §4).

**Resident-K verdict:** the residency *depth* is indeed 1 k_chunk (k-outer is correct on that — only one K chunk
is live at a time, same as today). The blocker is the **3-deep CB + per-step compute pop**, which makes a naive
"push once" deadlock or corrupt. Reader-side feasibility: **feasible, ~30-50 line swap + new per-q "first active
k_chunk" tracker (+ option-B scratch), but NOT landable reader-only.**

---

## 2. Compute side (`compute_streaming.hpp`, `sdpa_ring_v2`)

### Online-softmax state machine (today)
`AccumulatorHalf {sum, max, out}` (L67). Two physical halves `cb_*_A/B` (`ring_joint_sdpa.cpp` L118-123) wrapped
into a single `acc_state = {prev, cur}` (L147-150). Inside the K loop: `cur` = chunk being computed (writes
`cur.max` via eltwise-max vs `prev.max`, `cur.sum`, `cur.out`); `prev` = accumulated state read by SALAD. **The
swap is at L1986-1991** (`std::swap(q_prev, q_cur)`; non-ring path L1486). Only **one Q's accumulators are live at
a time**, double-buffered as A/B; the Q loop is the *outer* loop (L1707). **Confirmed.**

### Reorders cleanly:
- **Causal geometry is stateless per (k, q).** `q_start_tile` (L1718/1723), `causal_k_limit` (L1719), diagonal
  narrowing (L1877-85), straddle (L1917-28) all depend only on (q, k_chunk) + static layout. Only the accumulators
  and the `KV_chunks_processed` counters are threaded across the K loop → just make the counters per-Q (3 ints).
  **Reorder-safe.**
- **DST budget is fine.** Accumulators live in **L1 CBs, not DST registers**; 3 live accumulators do not touch the
  8/16-tile DST limit. The cost is L1 capacity (§3), not DST.

### The real compute difficulties:
- **3 live accumulators + SALAD-not-in-place.** k-outer needs `q_per_core`=3 independent `{out(16t), max(1),
  sum(1)}` live across the K loop. But `salad_correct_fused` (L471-544) reads `out_in_cb`(=prev.out) and writes a
  *different* `out_out_cb` (L510→L520) — **not in place**. So per-Q you need either **2×q_per_core** output buffers
  (6×16 tiles) or an **in-place SALAD rewrite** (new, unproven path). This is the single hardest sub-problem and is
  an L1 + buffer-duplication question.
- **Streaming normalize / writer handshake becomes q-interleaved.** Normalize fires per-Q at that Q's last K
  (`is_last_k = KV_chunks_processed[q]==per_q_valid_kv[q]`, L1779). Because `causal_k_limit` **grows with q**
  (L1719), different Q's finish their K-range at **different outer k_chunks**. The mechanism survives — but the
  compute→writer `cb_signal` "start draining `cb_out` row-by-row" handshake (L1785-88) carries no q-index today, so
  it must be made q-aware → **pulls the writer in (3-sided).**
- **`restore_from_staging` interaction.** The existing q_per_core>1 path *spills accumulators to DRAM between ring
  iters* (L1755-62, 1888-95, 2007-16) precisely because it can't keep all per-Q accumulators in L1 today. The
  k-outer reorder trades that DRAM spill for L1-resident per-Q accumulators; the two share the A/B + staging CBs and
  must be reworked together.

**Compute verdict:** control flow and causal masking reorder cleanly; the hard part is keeping `q_per_core` live
accumulators given SALAD's not-in-place write, plus making the writer drain q-aware. **Feasible, large, gated by
the accuracy test.**

---

## 3. L1 budget (independently recomputed from `ring_joint_sdpa_program_factory.cpp`)

Dtypes confirmed from source: `q=bf16` (2048 B/tile), `kv=bf8_b` (1088 B), out_im/accumulators = `Float16_b`
(2048 B), out=bf16. L1 = 1,572,864 B.

### Full CB table (matches prior doc exactly)
| CB | tiles | dtype | bytes |
|---|--:|---|--:|
| cb_q_in | 36 | bf16 | 73,728 |
| **cb_k_in** | **1080** | **bf8_b** | **1,175,040** |
| cb_v_in | 0 | — | 0 (aliases cb_k_in, latent V) |
| cb_qk_im | 20 | bf16 | 40,960 |
| cb_out_im_A / B | 16 / 16 | bf16 | 32,768 / 32,768 |
| cb_out | 16 | bf16 | 32,768 |
| cb_prev_out | 16 | bf16 | 32,768 |
| cb_max_A/B, cb_sum_A/B | 1+1+1+1 | bf16 | 8,192 |
| mask/scale/identity/stats/exp_max_diff/streaming | … | bf16 | ~17,488 |
| **TOTAL** | | | **1,451,536 (= prior doc, diff 0 B)** |

- **Free L1 = 121,328 B = 118.5 KB.** cb_k_in = **81.0 %** of all CBs. (All prior-doc numbers reproduce.)
- **K-CB depth (L356) is hard-coded** `(v_shares_k_buffer ? 3 : 2)` — **there is no `CHUNKED_K_CB_DEPTH` env knob
  in the current factory** (the prior doc's knob was added then reverted).

### Accumulator cost for the reorder
out_im is **Float16_b = 2048 B**:
- **Ping-pong (3q × A+B):** 96 out_im tiles, Δ +64 tiles = **+131,072 B** out_im; +stats ≈ **+147,456 B total**.
- **Single-buffered (3q):** 48 out_im tiles, Δ +16 tiles = **+32,768 B**; +stats ≈ **+36,864 B total**.

### K-CB depth 3→2 reclaim
One depth entry = Sk·DHt = 360 tiles × 1088 B = **391,680 B = 382.5 KB** freed.

### Net L1 verdict — **the correction**
| Scenario | net free L1 |
|---|--:|
| ping-pong accumulators, **no** reclaim | **−26,128 B → OOM** (this is the prior doc's "boxed" case) |
| single-buffered + depth-3→2 reclaim | 121,328 + 391,680 − 36,864 = **+476,144 B (465 KB) — FITS** |
| ping-pong + depth-3→2 reclaim | 121,328 + 391,680 − 147,456 = **+365,552 B (357 KB) — FITS** |

So **L1 is comfortable once the depth reclaim is credited** — even the ping-pong variant fits. The prior doc's
OOM was computed without applying the reclaim against the accumulators in the same breath.

**The remaining L1-adjacent risk is behavioral, not capacity:** the prior doc's q64 experiment showed dropping
K-CB depth 3→2 cost −9 pt in the *q-outer* order because V-remat could no longer overlap compute. The reorder's
premise (the prior doc's own §plan) is that in k-outer order compute spends ~3× longer per k_chunk, so the 3rd
K-slot becomes droppable. **That must be measured, not assumed.** If V-remat still needs the 3rd slot, use the
single-buffered accumulators *without* the reclaim — which does OOM — so this is the one place the reorder could
still get L1-squeezed. Mitigations: single-buffered + a modest k_chunk reduction, or the V^T-from-K^T compute
sourcing change (see companion doc) which removes V-remat entirely and moots the 3rd slot.

Also note CBs the prior doc's accounting glossed: `cb_qk_im` (40,960 B, 2nd largest, does *not* scale with
q_per_core), and `cb_prev_out`/`cb_out` (32,768 B each) which a 3-live-Q reorder may need to grow by up to
+65,536 B — still well inside the post-reclaim margin.

---

## 4. Chain / mcast protocol + fused-op coupling (`chain_link.hpp`, factory)

### The K mcast is a 1-hop row broadcast (simpler than "store-and-forward")
Three L1 semaphores per chain (factory `ChainSemaphores::create`):
- **sender_sem** (init 0, on injector): receivers `inc` it; injector `wait(==num_receivers)` then resets. The
  `MC_SEMWAIT` rendezvous.
- **receiver_sem** (init 0, on each receiver): injector mcasts `VALID` into it. The "K arrived" flag.
- **valid_sem** (const VALID): the payload word mcast into every receiver_sem.

Per K chunk: each receiver does exactly **one** `inc(injector.sender_sem)` + **one** `wait(own receiver_sem)`;
the injector does **one** `wait(sender_sem==N)` + reset + one data-mcast + one VALID-mcast (CL 210-214, 227-234).
**sender_sem is reset to 0 each forward and receiver_sem re-armed each receive — no cross-iteration accumulation.**
Balance is purely **forward-count == receive-count per (ring_iter, k_chunk), at injector and every receiver.**

### Role of q_iter today
`should_forward(…, q_iter_local)` gates `q_iter_local >= next_core_q_chunks`. For K-mcast,
`next_core_q_chunks == chain_max_q == row_max_q == loop_q_count`, so the gate is **never true** in the mcast path —
every one of the `max_q_per_core` iters forwards/receives. ⇒ today K is forwarded **once per (q_iter, k_chunk)** —
the 3× redundancy. Cutting to once-per-k_chunk changes the total forward count by the q_per_core factor and
**requires symmetric change on injector AND every receiver.**

### Where a hang originates
- **Asymmetric hoist.** If `forward()` is hoisted out of the q-loop but `receive()` is not (or vice versa), counts
  diverge → an over-receiving core hangs on a VALID that never comes; an under-receiving row hangs the injector's
  `wait(sender_sem==N)`. ⇒ the receive must be hoisted too, K held resident across the q-inner loop — the same
  2-sided requirement as §1.
- **The `next_core_q_chunks` gate** becomes meaningless (no q dimension) and must be replaced by a **k_chunk bound
  identical on injector and all receivers**, derived from a **row-wide union** of the K chunks any Q needs (NOT the
  per-Q causal limit). The injector must remain the row's max-work core and forward every k_chunk any receiver
  needs, or a receiver hangs.
- **Padded-iter uniformity moves from the q axis to the k axis.** Today padded iters (q_iter ≥ q_per_core) still
  participate in the mcast handshake to keep cores lock-stepped (only `cb_push_back`/Q-read are skipped). Under
  k-outer, the quantity that must be uniform per row is the **K-chunk forward count**; `iter_num_kv_chunks` is
  already derived identically per ring_iter, but the causal/balanced halving (RD 404-411) and the
  `dummy_kv_chunks` `%3`/`%2` phase alignment (keyed on `KV_chunks_processed_in_iter`, which the reorder reduces
  ~3×) must be re-derived in lock-step on both reader and compute.

### The dominant residual risk: fused-op / AllGather positional coupling
The SDPA op shares, with the fused AllGather op:
1. **One ordered `desc.semaphores` ID pool** — fused sems pushed first, then chain sems, and the AllGather helper
   allocates more "starting at `desc.semaphores.size()`". Kernels read sems by **compile-time index offsets**. Add
   /remove/reorder any semaphore → every downstream ID shifts → kernel reads the wrong sem → hang.
2. **One positional RT-arg stream** appended in fixed order (tensor addrs → head_chain → batch_chain →
   v_batch_chain → fused-op signaler args) across reader/writer/compute. Change the chain arg count → the fused-op
   receiver reads from the wrong offset → waits on the wrong AllGather signal → hang.
3. **One CB-index space** read by constexpr offset.

This is the exact failure mode that hung the prior K-mcast-to-writer attempt (**commit `d33553875fa`**): the K
mcast itself was ruled out; the **arg/CB/sem-layout perturbation** was the culprit.

**Mandatory de-risking:**
1. **Do not add/remove/reorder any semaphore, RT arg, or CB** without re-auditing every constexpr offset
   (reader L107-168) and the AllGather helper's `desc.semaphores.size()`-relative alloc. Prefer **reusing** the
   existing batch-chain sems and **repurposing** the already-plumbed `next_core_q_chunks`/`k_chain_max_q` for the
   new k-forward bound rather than adding args.
2. **Gate symmetrically + assert.** Replace the q-gate with a single k-chunk bound computed identically on
   injector and receivers; host-side `TT_FATAL` that forward-count == planned receive-count per row.
3. **Keep the injector = row-max-work core invariant** and forward the causal/balanced *union* of k_chunks.
4. **Validate behind the PCC test first.** Start from the proven safe timing-only ablation (skip the redundant
   read, output garbage but timing valid) to re-confirm the +pt, *then* do the risky 2-sided semaphore restructure.
5. **Watch for hang on first run** (per CLAUDE.md — if pytest output stalls, suspect the chain). Bisect by
   reverting to per-q forwards while keeping the loop reorder, to separate mcast-count asymmetry from fused-op
   layout drift.

---

## 5. Synthesis — is it possible, and what does it cost?

**Yes, it is possible.** No sub-problem is a true wall:
- L1 fits comfortably after the depth-3→2 reclaim (§3) — the prior "L1-boxed" verdict was premature.
- Causal masking and the online-softmax math reorder cleanly (§2); DST is untouched.
- The mcast rebalance is local and tractable (1-hop, reset-per-chunk) (§4).

**But it is a coordinated, hang-prone, multi-session change**, gated by four things that must move together:
1. **2-sided pop-cadence contract** (§1, §2): K received+pushed once, consumed by all 3 Q's before pop → reader CB
   semantics + compute pop logic change in lock-step.
2. **SALAD not-in-place** (§2): per-Q accumulation needs a 2nd buffer per Q or an in-place SALAD rewrite.
3. **q-aware writer handshake** (§2): the `cb_signal`/`cb_out` drain becomes interleaved across Q's → 3-sided.
4. **Fused-op layout discipline** (§4): the proven `d33553875fa` hang vector — no semaphore/arg/CB churn without a
   full offset re-audit.

### Recommended sequencing (lowest-risk path to the +0.19 ms)
1. **Re-confirm the prize** with the existing safe timing-only ablation (skip redundant K read, gate on q_iter==0;
   output garbage, timing valid). Already done in the companion doc: **2.999 ms / 63.2 %**.
2. **Settle the V-remat-vs-K-CB-depth question empirically** in k-outer order: does the 3rd K-slot become droppable
   when compute runs ~3× longer per k_chunk? This decides single-buffered-without-reclaim (OOM) vs
   single-buffered-with-reclaim (465 KB margin). Consider doing the **V^T-from-K^T compute sourcing** change first
   (companion doc, §"the one path to the −0.212 ms") — it eliminates V-remat, removes the 3rd-slot dependency, and
   is independently a larger measured win.
3. **Land the reorder behind `chunked_enabled && !is_balanced && NHK==1`** so non-chunked/balanced/joint paths are
   untouched. Single-buffered per-Q accumulators (delicate Phase-2 in-place rescale) + K-CB depth 2.
4. **Validate to PCC ≥ 0.99** (unset `CHUNKED_SKIP_PCC`); expect several device-reset/hang cycles. Bisect hangs
   per §4.

### Net correction to the companion doc
- ✅ Redundant-K read is the prize; k-stationary reorder is the principled fix; mechanism is contention/traffic
  (download‖mcast won't help). **Confirmed.**
- ✅ It's a 2-sided (really 2-to-3-sided) reader↔compute(↔writer) restructure, hang-prone. **Confirmed and detailed.**
- ⚠️ **"L1-blocked / genuinely boxed in" — overstated.** L1 fits with 357–465 KB margin after the depth reclaim;
  the residual L1 risk is the *behavioral* V-remat/3rd-slot question, not capacity. The reorder is gated by
  **engineering effort + hang-risk + the fused-op coupling**, not by L1 bytes.

---

## Appendix — key code locations
| concern | file | lines |
|---|---|---|
| reader loop nest | `dataflow/ring_joint_reader.cpp` | 359 / 418 / 463 |
| K slice (q-invariant) | `…/ring_joint_reader.cpp` | 484-507 |
| K receive/fetch/forward | `…/ring_joint_reader.cpp` | 519-538 |
| K push, Q-read, V-remat | `…/ring_joint_reader.cpp` | 575 / 582-616 / 618-642 |
| dummy_kv_chunks drain | `…/ring_joint_reader.cpp` | 673-684 |
| compute loop + accumulators | `compute/compute_streaming.hpp` `sdpa_ring_v2` | 1707 / 1766 |
| prev/cur swap | `…/compute_streaming.hpp` | 1986-1991 (non-ring 1486) |
| SALAD (not in-place) | `…/compute_streaming.hpp` `salad_correct_fused` | 471-544 (510, 520) |
| causal limit per (k,q) | `…/compute_streaming.hpp` | 1718-1719 / 1877-1885 |
| streaming normalize / cb_signal | `…/compute_streaming.hpp` | 1779-1788 / 1063-1075 |
| dummy-pop / phase align | `…/compute_streaming.hpp` | 2020-2032 |
| K-CB depth (hard-coded 3) | `device/ring_joint_sdpa_program_factory.cpp` | 356 |
| CB allocation | `…/ring_joint_sdpa_program_factory.cpp` | 731-793 |
| mcast injector / balance | `…/ring_joint_sdpa_program_factory.cpp` | 1252-1367 |
| shared sem/arg/CB layout | `…/ring_joint_sdpa_program_factory.cpp` | 295-326 / 531-587 / 1505-1551 |
| ChainLink protocol | `dataflow/chain_link.hpp` | 186-243 |
| prior fused-op hang | commit | `d33553875fa` |
