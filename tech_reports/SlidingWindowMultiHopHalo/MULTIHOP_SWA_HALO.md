# Multi-hop SWA halo: unlocking 4k and 2k prefill chunks for Gemma4

**Goal:** remove the `chunk >= sliding_window * CP` (= 8192 at W1024/CP8) floor so short prompts
stop paying a full 8192-token chunk, and measure the TTFT win at 4k (and then 2k).

**Status:** **chunk 4096 is done: 1.40x better TTFT and numerically gated against torch (PCC 0.9997).**
Chunk 2048 needs h=4 hops and the fabric offers only 2 links, so it is blocked on one scoped
refactor (§10.3). See §8–§10.
**Branch:** `main` @ `df15dfd17d5` (rebuilt clean in `/data/kmabee/tt-metal-2`)
**Machine:** `bh-glx-120-b03u02` — BH Galaxy, 32 ASICs
**Date:** 2026-09-16

---

## 1. What the constraint actually is

`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:618`

```cpp
TT_FATAL(halo_tokens <= N_local_q,
  "Chunked sliding halo {} (window {}) exceeds the per-device Q slab {}; wider windows need a "
  "multi-hop halo", halo_tokens, window_size, N_local_q);
```

* `halo_tokens = ceil((W - 1) / k_chunk) * k_chunk` — the window rounded up to whole K chunks.
  For Gemma4: W=1024, k_chunk=128 → **halo = 1024 tokens**, *independent of chunk size*.
* `N_local_q = chunk / CP` — the per-device Q slab. At CP=8: 1024 tokens at chunk 8192.

So today the halo must fit in **one** predecessor slab — a single hop around the CP ring.
Gemma4 sits exactly on the boundary (`halo_tile_rows == q_local_tile_rows == 32`), which is the
coincidence the branch author flagged as having hidden shape-specific bugs before.

Hops needed, `h = ceil(halo_tokens / N_local_q)`:

| chunk | N_local_q (CP=8) | h | status |
|---:|---:|---:|---|
| 8192 | 1024 | 1 | works today |
| 4096 | 512 | **2** | needs multi-hop |
| 2048 | 256 | **4** | needs multi-hop |
| 1024 | 128 | 8 | needs multi-hop + the hop-8 source is *local* (see §5) |

Everything else in the op's sliding allowlist already passes at 4096 and 2048: `ring_size ∈ {4,8}`,
`q_chunk ∈ {64,128}`, `k_chunk == 128`, `N_local_q % q_chunk == 0`, `N_local_q % k_chunk == 0`
(512 and 256 are both multiples of 128).

**The model side needs almost nothing.** `models/demos/gemma4_d_p/tt/attention/ring_prefill.py:385`
sizes the compact gathered buffer as `ceil((W-1)/k_chunk)*k_chunk` — the *window*, not the chunk.
That number does not move when the chunk shrinks: the same 1024 tokens of halo simply arrive as
`h` smaller pieces from `h` different neighbours. Confirms the teammate's expectation that this is
kernel-side plus a chunk-size constant.

---

## 2. How the halo works today (and why multi-hop is a natural extension)

Three cooperating pieces, all already on `main`:

1. **Host layout** — `sliding_halo_layout.{hpp,cpp}` +
   `kernels/sliding_window_work_plan.hpp::chunked_sliding_halo_source_start_tile()`.
   Device `s` sends the **last `halo_tile_rows` rows of its own slab** to `s+1`. If `s+1 == ring_size`
   the receiver is device 0 of the *next* group, so the payload is taken from group `current-1`.
2. **Transport** — `ring_attention_neighbor_halo_exchange_helper()` in the ring-attention
   all-gather program factory, invoked from `ring_joint_sdpa_program_factory.cpp:2926`. A reader
   kernel pulls the tail from the local KV cache into a CB; a writer kernel unicasts it over fabric
   into the neighbour's compact buffer at `dest_page = src_page - origin` (so the halo lands at row 0),
   and fused-atomic-incs `semaphores[2]` on the last packet. The receiving device's halo reader waits
   for that semaphore, signals the SDPA cores, and decrements.
3. **Consumption** — `build_sliding_q_work_plan()` (runs *on device*, in `ring_joint_reader.cpp:844`)
   produces per-K-chunk source ranges; the reader uses `compact_k_chunk * Sk_chunk_t` as the row
   index whenever the source is not the local rank (`ring_joint_reader.cpp:928`).

Two facts make multi-hop tractable:

* **The SDPA reader is source-agnostic.** It never computes a hop distance — it reads whatever
  `compact_k_chunk` the work plan hands it. Getting the plan's compact offsets right is the whole job
  on the consumption side.
* **The transport already does multi-hop unicast.** The linear-topology wrap case
  (`ring_joint_sdpa_program_factory.cpp:2909`) sets `unicast_hops = ring_size - 1` and sends
  *backward* so device N-1 can reach device 0. So "send my tail `d` hops away" is an existing,
  exercised code path, not new fabric work.

---

## 3. Design: h parallel unicasts into disjoint rows of the same compact buffer

For receiver `r`, hop `d ∈ [1, h]`, source `s = (r - d) mod R`:

```
count(d)        = min(q_local, halo_rows - (d-1)*q_local)   # rows this hop carries
src tail start  = source_slab_base + q_local - count(d)     # always a TAIL of the source slab
dest row base   = halo_rows - d*q_local   (clamped at 0)    # oldest hop lands at row 0
source group    = (s > r) ? current_group - 1 : current_group
direction       = (s > r) ? backward (r + R - s hops) : forward (d hops)
```

The compact buffer keeps its current size (`halo_rows`, = 1024 tokens for Gemma4) and its current
meaning — the `halo_rows` tokens immediately preceding the receiver's slab, oldest first. At h=1 every
formula above collapses to exactly today's behaviour, which is the regression argument.

Sync: each hop's writer fused-atomic-incs the **same** `semaphores[2]`, each hop's reader consumes one
arrival and signals the SDPA once, and `forward_writes_expected` goes `1 → h`
(`ring_joint_sdpa_program_factory.cpp:431`). The SDPA reader already waits for
`1 + expected[0] + expected[1]` signals *before* its K loop
(`ring_joint_reader.cpp:690`), so it blocks until **all** h payloads have landed. Arrival order does
not matter because the count is what gates, and the sliding path ignores the ring-id the signal carries.

### Rejected alternatives

* **Relay/chain** (i forwards to i+1, which forwards on): serializes h fabric transfers and needs an
  extra sync round inside the op. The h direct unicasts all fly in parallel.
* **Fat single hop**: impossible — the predecessor does not hold its own predecessor's KV shard.
* **Treat sliding layers as dense** (gather the whole prefix): throws away the entire point of SWA.

---

## 4. Modelled estimate — how much TTFT this buys

Using the cost model fitted on this box (`~/debug-docs/gemma4_prefill_chunk_scaling-noissue/`),
whose out-of-sample predictions held to ≤2.3% across an 8× context and 16× chunk-count range:

```
T(n, C) = a(C) + n * slope(C)
a(C)     = rho*C^2 + alpha*C + K      rho=1.845e-7 ms/tok^2, alpha=17.92 us/tok, K=76.25 ms
slope(C) = beta*C^2 + gamma*C         beta=1.217e-7 ms/tok^2, gamma=0.503 us/tok
```

**`K = 76.25 ms` of per-chunk fixed cost is the TTFT floor**, and it is what makes small chunks pay:

| chunk C | a(C) = single-chunk device time | of which fixed K |
|---:|---:|---:|
| 8192 | 235.4 ms *(measured: 235.1)* | 32% |
| 4096 | **152.8 ms** | 50% |
| 2048 | **113.7 ms** naive → **~124 ms** corrected (§4.1) | 62–67% |
| 1024 | 94.8 naive → ~115 corrected | 66–80% |

### 4.1 One correction the pure fit misses: Q-chunk quantization

SDPA splits work as `B * NH * num_q_chunks` units over 120 cores; wall time is
`ceil(units / 120)` units deep. For Gemma4 at CP8/TP4 (NH=8 Q heads/device, q_chunk=64):

| chunk | units/device | depth | depth per token | vs 8192 |
|---:|---:|---:|---:|---:|
| 8192 | 128 | 2 | 2.44e-4 | 1.00 |
| **4096** | 64 | **1** | 2.44e-4 | **1.00 — identical** |
| 2048 | 32 | 1 | 4.88e-4 | 2.00 — twice the cost per token |

So the downward extrapolation to **4096 is safe** — the SDPA term keeps exactly the same per-token
efficiency, which is the assumption the linear `alpha` term encodes. At **2048** the work no longer
fills the grid, so the SDPA share of `alpha` (~27% of it, from the op profile) stops halving. That is
the +10 ms correction above, and it is why 2048 is the practical floor rather than 1024.

### 4.2 TTFT by prompt length

`TTFT(ISL, C) = N*a(C) + slope(C)*N*(N-1)/2`, `N = ceil(ISL/C)`. Device time, staging excluded
(~1–10 ms/chunk on top, roughly chunk-independent):

| ISL | chunk 8192 (today) | chunk 4096 | chunk 2048 | best |
|---:|---:|---:|---:|---|
| 1024 | 235 ms | **153 ms (1.54x)** | **124 ms (1.90x)** | 2048 |
| 2048 | 235 ms | **153 ms (1.54x)** | **124 ms (1.90x)** | 2048 |
| 4096 | 235 ms | **153 ms (1.54x)** | 249 ms (0.94x) | 4096 |
| 8192 | 235 ms | 310 ms (0.76x) | 500 ms | 8192 |

**The rule is "use the smallest legal chunk ≥ ISL".** The win is entirely the padding a short prompt
would otherwise pay, so it is bounded by `a(C)` and saturates: 1.54x at 4k, 1.90x at 2k, ~2.0x at 1k.
Long-context throughput is untouched — 32768 remains the optimum there (+24.3% over 8192, measured).

### 4.3 What multi-hop itself costs

Halo bytes are **unchanged** — 1024 tokens × 4 local KV heads × 256 dim × 2 (K,V) ≈ 2.2 MB per sliding
layer per device, whether it comes as one message or `h`. Only the message count rises, so the extra
cost is `h-1` extra packet setups + semaphore round-trips per sliding layer (~tens of µs per chunk
total). Modelled impact on `K`: **< 1 ms**, i.e. inside the noise of the numbers above.

The real resource constraint is **CCL worker cores**: each hop takes `num_links` workers (2 on BH) from
the single reserved CCL column (10 cores), so `h ≤ 5` at num_links=2. h=2 (chunk 4096) uses 4 of 10;
h=4 (chunk 2048) uses 8 of 10. Tight but feasible — and a reason 2048 is the floor.

---

## 5. Code scope

| # | File | Change |
|---|---|---|
| 1 | `sdpa/device/kernels/sliding_window_work_plan.hpp` | `max_source_ranges` 2 → 9; `chunked_sliding_halo_source_start_tile()` takes the receiver (or hop distance) and drops `halo > q_local` bailout; per-hop compact base in `first_compact_k_chunk`; `build_sliding_q_work_plan()` loses its `halo_tile_rows > q_local_tile_rows` early-out |
| 2 | `sdpa/device/sliding_halo_layout.{hpp,cpp}` | carry `hop_count`; `send_tail_start_tile(source, hop)`; per-hop `count`/`dest_base` accessors |
| 3 | `ccl/ring_attention_all_gather_async/.../program_factory.{hpp,cpp}` | `RingAttentionNeighborHaloConfig`: add `dest_row_base`, per-hop `send_to_next_count_Ht`; writer dest page += `dest_row_base * Wt` |
| 4 | `.../kernels/ring_attention_neighbor_halo_writer.cpp` | apply `dest_row_base` to the destination page id |
| 5 | `.../kernels/ring_attention_all_gather_metadata.hpp` | `compute_halo_tail_start_Ht()` — the on-device (trace-safe) twin of #1; must take the hop and drop the same bailout. **Gemma4 runs the metadata path, so this one is mandatory, not optional.** |
| 6 | `sdpa/device/ring_joint_sdpa_program_factory.cpp` | loop the halo helper over `d = 1..h` with per-hop coord/direction/`unicast_hops`/`core_grid_offset`; `forward_writes_expected = h` |
| 7 | `sdpa/device/ring_joint_sdpa_device_operation.cpp:618` | relax `halo <= N_local_q` to `h <= ring_size` (+ a CCL-core budget check) |
| 8 | `models/demos/gemma4_d_p/demo/text_demo_prefill.py:197` | drop/relax the `chunk >= window*cp` skip |

Danger spots called out in the code itself: the host layout (#1), the device metadata twin (#5) and the
reader must agree exactly, or "cb_output page counts drift" → silent hang. Every one of them is exercised
by the same three formulas in §3, so they get one shared helper rather than three copies.

## 6. Test ladder (cheapest first)

1. **Host gtest, no device** — `tests/ttnn/unit_tests/gtests/sdpa/test_sliding_window_work_plan.cpp`
   already covers the 1-hop geometry (6 tests). Add h=2/h=4 cases: ranges cover the window with no gap
   or overlap, compact offsets are disjoint and in range, h=1 output is byte-identical to today.
2. **Single-op device test with a torch reference** —
   `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_gemma_complete_group_sliding_geometry`
   is *exactly* Gemma's geometry (D256, 4Q:2K, W1024, `chunk_size_local=1024`). Add
   `chunk_size_local=512` (h=2) and `256` (h=4) variants. Runs on the galaxy at SP=4/TP=8 with **no
   62 GB model load** — this is the fast correctness loop, and it is the only numerical gate that exists
   (the end-to-end demo asserts finiteness only). Note SP=4 caps coverage at h≤3.
3. **End-to-end Gemma4** — `text_demo_prefill.py::test_prefill_long_context_traced` at
   `ctx_32k-chunk4096-8x4`, then the short-ISL TTFT comparison that is the actual deliverable.

## 7. Risks

| risk | mitigation |
|---|---|
| Host/device/reader formula drift → **silent hang**, not an error | one shared constexpr helper, used by all three; gtest pins the h=1 output |
| CCL worker cores exhausted at h=4 | explicit host check; 2048 may need num_links=1 for the halo |
| Q-chunk quantization makes 2048 worse than modelled | measured in step 3; 4096 is unaffected (§4.1) |
| Wrap-group clipping at group 0 for the farther hops | the existing "clamp origin, preserve length" trick generalizes; gtest case for group 0 at h=2/4 |
| A mesh-mismatched weight cache deadlocks in layer 0 | use `/data/kmabee/hf_cache/.../tensor_cache_bf16_mesh8x4`; verify no "reusing legacy" line |


---

# 8. Results (session of 2026-09-16)

## 8.1 Baseline reproduced from `main`

`main` @ `df15dfd17d5`, rebuilt clean, mesh-matched cache (`tensor_cache_bf16_mesh8x4`, no legacy fallback):

```
ctx_32k chunk8192 8x4 : DEVICE 32768 tok in 1.0s (31,412 tok/s), chunk0 = 243.1 ms
```

**243 ms is the TTFT any prompt of ≤8192 tokens pays today** — the number chunk 4096 has to beat.
It reproduces the archived 32k/8192 figure (1.00 s / 32,445 tok/s) to ~3%, so `main` carries the
prefill work at the same performance as the branch it came from.

## 8.2 What is implemented

All of §5, plus one thing the scope missed: a per-hop **fabric link** and a per-hop **signal
semaphore** (§8.4). 14 files, ~405 insertions. `git diff --stat` in `/data/kmabee/tt-metal-2`.

## 8.3 What is verified

| check | result |
|---|---|
| Work-plan math, 660 geometries incl. h=2/4/8, rings 2/4/8, bounded + unbounded caches | **clean** — every window row covered exactly once, compact offsets disjoint and in range, verified against an independent reconstruction of each row's absolute position |
| Exact Gemma device geometry (SP4, W1024, 512-tok slab, k_chunk 128, **q_chunk 64**) | **clean** |
| One-hop pinned values (the existing gtest's `PinnedDevice0Geometry`) | **unchanged** (static_assert) |
| One-hop end-to-end on device, Gemma geometry vs torch | **PASS**, PCC 0.99972 / 0.99973 — the deployed 8k path is not disturbed |
| Two-hop halo transport on device | **works**: both hops ship, route `{mesh 0, hops 1}` and `{0, 2}`, compact offsets 128 and 0 (i.e. rows 16–31 and 0–15 of a 32-row halo — exactly as designed), all 8 exchanges complete on 4 devices |
| Two-hop halo→SDPA sync | **works** after §8.4 — every device reaches and passes `consume_halo_signals` |
| Two-hop numerical result | **not reached** — hangs after the sync, see §8.5 |

Host-only harness: `scratchpad/plan_check.cpp`, `exact_check.cpp` (compile with clang++ against the
header, no device, ~2 s). Keep this loop — it caught nothing *wrong*, but it is what makes the
device failures interpretable.

## 8.4 Two real bugs found, both fixed

**1. Two exchanges cannot share a fabric link.** One worker per (link, direction) owns an EDM
channel. With both hops' writers on link 0 forward, hop 1 opened the connection, sent, and then
stalled mid-transfer — the reader filled the CB behind it and the whole op wedged. The convention is
visible in `all_gather_async` (one core per link *per direction*) and in
`reduce_scatter_minimal_direct` ("one worker core per fabric link, owning that link's forward AND
backward connection"). Fixed by giving each hop its own link (`link_base`), which also keeps
aggregate bandwidth constant since each hop carries 1/h of the same halo. **This caps h at
`num_links` (2 on BH)** — enough for chunk 4096, not for 2048.

**2. Two workers must not signal the same fused-op semaphore.** Each hop's halo reader signalled the
SDPA cores on the same semaphore; `Semaphore::up` is a NoC atomic increment whose own header warns
"*atomicity is not guaranteed on WH/BH, multiple cores incrementing simultaneously may lead to lost
updates*". Symptom: exactly one device of four stuck waiting for a second signal that had already
been sent — timing-dependent, which is what a lost update looks like. Fixed by giving hop 1 and hop 2
the signaler's two per-direction semaphores (`push_all_gather_fused_op_rt_args(..., hop - 1)`) so each
has a single incrementer. **This is the second thing capping h at 2** (only two such semaphores exist).

A third change was made on a wrong hypothesis and is worth keeping anyway: the sliding path now waits
for all halo arrivals in **one** decrement rather than one per arrival. `Semaphore<>` is
`LOCAL_NONATOMIC`, so `down()` is a plain read-modify-write that can swallow an increment landing
mid-update; with one arrival that never mattered.

## 8.5 The remaining blocker

With both fixes in, every device now passes the halo sync (`SDPAR consumed` on all four) and then the
op still hangs — host parked in `completion_queue_wait_front`. So the failure moved out of the
transport/sync and into the **K loop or compute**: reader and compute each build the work plan
independently (`ring_joint_reader.cpp:844`, `compute_streaming.hpp:2525`) and must agree
chunk-for-chunk on how many K chunks are pushed and consumed. With three source ranges instead of
two, the natural suspects are the compute-side K-loop bound and the lightweight-mask geometry, not
the plan itself (which is verified host-side for exactly this geometry).

**Next step, concretely:** DPRINT `total_k_chunk_count` and the per-work-item
`(source_ring_id, source_k_chunk, compact_k_chunk)` from *both* the reader and the compute kernel on
one core of the device whose Q actually reaches two slabs back (ring index ≥ 2 — note that on this
mesh the DPRINT chip prefix is NOT the ring index: chips 0–3 are TP positions inside SP row 0, which
is why the first traces all showed `ranges=1`). If the two disagree, the mismatch is the hang.

## 8.6 Cost of the debugging loop, for whoever picks this up

- The **watcher is unusable on this op**: `TT_METAL_WATCHER=10` fails it with "Program size (29168)
  too large for kernel config buffer (26624) on ACTIVE_ETH" — and it fails the *unmodified* one-hop
  test identically, so it is a watcher artifact, not a symptom. Don't chase it.
- DPRINT is the tool that works. Current API is printf-style — `DPRINT("x={}\n", v)`, include
  `api/debug/dprint.h` (not `debug/dprint.h`); the old `<<` form is a static_assert.
  `TT_METAL_DPRINT_CORES` needs the parenthesised form `"(11,0),(11,1)"` — `11,0-11,5` silently
  enables one core.
- Kernel-source edits DO invalidate the JIT cache (verified by a probe disappearing), so no cache
  clearing is needed; host edits force a rebuild anyway.
- `timeout --signal=INT` does **not** kill a device-hung pytest. Use `timeout -k 10 <s>` (SIGTERM
  then SIGKILL).
- Every hard kill needs `tt-smi -glx_reset` (~2 min) before the next run, per
  [[tt-galaxy-fabric-run-hygiene]].

## 8.7 Where the estimate stands

Unchanged and untested on hardware: chunk 4096 should give **~153 ms vs 243 ms measured at 8192 for
ISL ≤ 4096 (~1.5x)**. The two structural caps discovered above (`h ≤ num_links`, `h ≤ 2 signal
semaphores`) do not affect chunk 4096 (h=2) but do block chunk 2048 (h=4) until a design change:
one exchange per link that loops over hops internally, which needs one fabric connection and one
signal per worker regardless of h. That is the recommended shape for the 2k work.


---

# 9. RESULT: chunk 4096 works, 1.40x better TTFT

## 9.1 Measured, same box, same session, `main` + this change

| | chunk 8192 (today) | chunk 4096 (this work) | |
|---|---:|---:|---|
| **chunk-0 device time = TTFT for ISL ≤ chunk** | **243.1 ms** | **174.2 ms** | **1.40x** |
| chunk 1 (ring depth 1) | 254.7 ms | 177.3 ms | |
| 32k total, 4 vs 8 chunks | 1.0 s / 31,404 tok/s | 1.5 s / 22,164 tok/s | long-context throughput is worse, as expected |
| trace compile | ~120 s | 60.5 s | |

Both are `ctx_32k … 8x4`, `readback_final`, warm weight cache, and the 8192 row was re-measured
**with the multi-hop code in the build** — it is identical to the pre-change baseline (243.1 ms,
31,404 vs 31,412 tok/s), so nothing on the deployed path moved.

**A prompt of ≤4096 tokens now costs 174 ms instead of 243 ms.** The crossover rule from §4.2 holds:
at ISL 8192 two 4096-chunks cost 174.2 + 177.3 = 351 ms against one 8192-chunk's 243 ms, so the
chunk should track the prompt, not replace 8192 everywhere.

## 9.2 Model vs measurement

Predicted 152.8 ms, measured 174.2 ms — the model was **12% optimistic**, its first miss outside
±2.3%. The 21 ms gap is ~0.43 ms per sliding layer, which is about what the second exchange should
cost: one more fabric round trip and one more signal per layer, with each hop now on a single link
instead of both. So the per-chunk fixed term `K` grows with hop count; it is not a defect in the
chunk-size model, it is a cost multi-hop adds. Worth re-fitting `K(h)` if 2k is pursued.

## 9.3 The honest caveat

**The 4096 path has no numerical gate.** The end-to-end test asserts finiteness only, and the
op-level test that *does* compare against torch (`test_ring_joint_attention_gemma_two_hop_sliding_halo_geometry`,
added in this work) still hangs at SP4 after the halo sync — see §8.5. So 1.40x is a measured
performance number on a path whose correctness is **not** established. Do not ship it on that basis.

Note the split: the op test hangs at **SP4 on the scalar KV-pad-rotation path**, while the model runs
green at **CP8 on the trace-safe metadata path**. That difference is itself the best clue for the
remaining bug, and it is cheap to exploit — the two paths derive the halo tail start from different
code (`chunked_sliding_halo_source_start_tile` on the host vs `compute_halo_tail_start_Ht` on device),
and only the host one is exercised by the hanging test.

## 9.4 Ordered next steps

1. **Fix the SP4 op test** (§8.5) and get a PCC number for h=2. Without it nothing here is shippable.
   Start from the scalar-vs-metadata split in §9.3.
2. **Re-run 4096 at 256k** to confirm long-context throughput only degrades as modelled.
3. **Chunk 2048 (h=4)** needs the design change in §8.7: one exchange per link that loops over hops
   internally, so hop count stops consuming links and signal semaphores.
4. **Re-fit `K(h)`** once 2k runs, and re-derive the optimum chunk per ISL.

## 9.5 Guards that had to be relaxed (four copies of the same rule)

`chunk >= window * cp` is duplicated in four places, and each one fails differently:
`demo/text_demo_prefill.py:197` (skip), `tt/common.py:52` (ValueError), `tt/model.py:156`
(ValueError), and the op itself (`ring_joint_sdpa_device_operation.cpp:618`, TT_FATAL). All four now
express the same thing as a hop count. Worth collapsing into one helper upstream.


---

# 10. Session 2 (2026-09-16 pm): correctness for 4k, and what blocks 2k

## 10.1 4k is now numerically gated — PCC 0.9997 vs torch

The gap left open in §9.3 is closed. Two new op-level tests run on **SP8 + linear fabric, the exact
layout Gemma4 prefill uses**, against the existing torch reference:

| test | geometry | result |
|---|---|---|
| `..._chunked_sliding_linear_topology_accuracy` (existing, control) | 128-tok window over a 128-tok slab → **1 hop** | **PASS** |
| `..._multi_hop_sliding_halo_linear_topology_accuracy` (**new**) | 256-tok window over a 128-tok slab → **2 hops** | **PASS, PCC 0.99968 / 0.99965, RMSE 0.0062 / 0.0059** |
| `..._multi_hop_sliding_halo_linear_topology_accuracy` (**new**) | 512-tok window over a 128-tok slab → **4 hops** | refused with a clean error (§10.3) |

The two-hop case is the chunk-4096 Gemma4 geometry in miniature (1024-tok window over a 512-tok
slab). Making the *window* wider rather than the chunk smaller is what let this run at SP8 — the
chunk cannot shrink below `k_chunk * cp` without breaking `N_local_q % k_chunk == 0`.

The SP4 test from §8.5 still hangs, and that is now understood to be a **different branch**: SP4 runs
`Topology.Ring`, where a wrapping hop goes *forward through the physical wrap link*; SP8 runs
`Topology.Linear`, where it goes backward. The model uses Linear. The Ring branch is a real bug but
not one that affects Gemma4, and it is no longer in the way.

End-to-end re-verified after the protocol refactor below: **chunk 4096 chunk-0 = 173.8 ms**,
32k in 1.5 s, exit 0 (was 174.2/174.5 ms — reproducible to 0.4%).

## 10.2 A gate that did NOT work, and why it is worth knowing

The first attempt at a 4k correctness gate was to compare final-chunk hidden states between chunk
4096 and chunk 8192 on the same 32k prompt — chunking should not change a token's value. It gave
PCC 0.986 with a worst row of 0.485, which looks damning until you run the **control**: chunk 8192 vs
chunk **16384**, two *known-good* single-hop configurations, which gives PCC 0.992 with a worst row
of **0.456** — no better. Over 60 layers of bf16 activations and bfp8 KV, different chunk sizes simply
do not agree closely, so the comparison has no resolution for this question at full depth.

Keep the dump (`GEMMA4_PREFILL_DUMP_DIR`, `text_demo_prefill.py`) — it is a good gate for *gross*
corruption and it is what showed the 4k output is no further from 8192 than another known-good chunk
size is. But do not read a low cross-chunk PCC as a bug without running the control first. A sharper
version would cut the layer count (`create_tt_model(num_layers=...)`), which removes the error
accumulation — at the cost of a 62 GB checkpoint read, since that path skips the warm-cache branch.

## 10.3 What blocks 2k: four hops, two links

`chunk 2048 / CP8` gives a 256-token slab, so the 1024-token window needs **h = 4** hops. Each
concurrent hop needs its own fabric link (§8.4), and this Galaxy exposes **2**:

```
TT_FATAL: Chunked sliding halo needs 4 hops but only 2 fabric links are available;
          each concurrent hop requires its own link
```

The 2 is a fabric-topology property, not a shortage of wires: the runtime reports *"4 eth channels,
but only 2 routing planes are available"*, and the routing-plane count comes from the mesh graph's
declared link counts (`control_plane.cpp:185`), so raising it is a system-level change that would
affect every CCL — out of scope here.

**The fix is to stop giving each hop its own worker.** One worker per link should carry
`ceil(h / links)` hops *sequentially*: same bytes, same fabric channels, just serialized. Concretely:

* `hops_per_worker` becomes a compile-time arg; the reader and writer loop over per-hop runtime-arg
  blocks (`tile_start/end`, `in_origin`, `out_origin`) instead of a single one.
* The packet route (`unicast_route_arg0/1`) and `send_backward` move from compile-time to per-hop
  runtime args, since hops in one worker can differ in direction.
* The writer opens both fabric connections when the hops it owns disagree on direction.
* The host calls the helper **once** with a vector of per-hop configs instead of once per hop.

Nothing else changes: the work plan, compact-buffer layout, rendezvous semaphore and SDPA side
already handle any h. Estimated 1–1.5 h including builds. Expected cost at 2k: the halo's ~2–5 ms per
chunk roughly doubles in latency, against a modelled ~50 ms/chunk saving.

## 10.4 The protocol refactor that removed the signal-semaphore cap

§8.4 capped h at 2 because each hop signalled the SDPA on its own semaphore and only two exist. That
is gone: **every hop now delivers its ready-increment to one rendezvous worker core** (hop 1's), and
only that exchange waits — for all `h` arrivals — and signals the SDPA once. So there is exactly one
incrementer per semaphore (what `Semaphore::up` needs) and the SDPA side is back to the unmodified
single-consume it had on `main`. Signal semaphores no longer bound the hop count; only links do.

Re-verified after the refactor: 1-hop PASS, 2-hop PASS at the same PCC, model e2e at 4096 green.

---

# 11. Session 3 (2026-09-16 pm): hops share links, and the measured tradeoff

## 11.1 Link sharing shipped — by hand-off, not by kernel restructuring

§10.3 planned to restructure both halo kernels to carry several hops each (~1–1.5 h). That was not
needed. **The fabric already supports sequential reuse of an EDM sender channel**, and says so:
`open_start()` reads "the cursor block left by the *previous producer on this channel*" and
`close_start()` persists it "for the *next connection* on this channel"
(`edm_fabric_worker_adapters.hpp`). Only CONCURRENT workers on one channel stall each other.

So hops hand the link over with a local semaphore instead:

* `RingAttentionNeighborHaloConfig` gains `waits_for_predecessor`, `signals_successor`,
  `chain_semaphore_id`, `successor_noc_x/y`.
* `ring_attention_neighbor_halo_writer.cpp` waits on that semaphore before
  `fabric_connection.open()` and increments its successor's after `close()` — after the close, not
  after the last send, because the producer cursor is persisted in `close_start()`.
* `ring_joint_sdpa_program_factory.cpp` drops the `hops <= links` TT_FATAL for
  `link = (hop-1) % min(hops, links)`, chaining hop h to hop h+span. One semaphore, allocated over
  the union of hop worker cores, id chosen to avoid the fabric's own later per-core allocations.

~50 lines, no arg-layout churn, and the queued hop overlaps its DRAM reads with its predecessor's
send — so only the fabric transfer serialises. Both chain flags are false whenever hops ≤ links, so
the 1- and 2-hop paths are untouched by construction.

**Gate:** 1-hop, 2-hop and 4-hop accuracy tests all PASS (69 s). 4-hop is 4 hops on 2 links.

## 11.2 Measured: the full chunk-size sweep on 8x4

One ctx-32k run per chunk size (`gemma4_runs/mh_sweep`, 9m53s, all 5 PASS). Per-chunk device times
are linear in chunk index, so `t_i = a + slope·i` yields both coefficients, and
`T(ISL,C) = N·a + slope·N(N−1)/2` extrapolates — validated against the previous session's
independent 256k runs to **0.9% (8192) and 0.6% (16384)**. This replaces four expensive 256k runs
with five cheap 32k ones.

| chunk | hops | TTFT = a | slope | T(256k) | vs 8192 TTFT | vs 8192 total |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 4 | 131.3 ms | 1.458 | 28.66 s | 1.85x better | 2.09x worse |
| 4096 | 2 | 174.2 ms | 2.995 | 17.19 s | 1.39x better | 1.25x worse |
| 8192 | 1 | 242.7 ms | 11.990 | 13.71 s | — | — |
| 16384 | 1 | 443.7 ms | 37.100 | 11.55 s | 0.55x | 0.84x |
| 32768 | 1 | 928.2 ms | (126.2) | 10.96 s | 0.26x | 0.80x |

Regression: 8192 → 242.7 ms vs 243.1 ms, 4096 → 174.2 vs 173.8. Deployed path unchanged.

## 11.3 The hop count costs almost nothing (~4-5 ms at chunk 4096)

First attempt, which was WRONG: fit the one-hop curve through the three h=1 points
(`a(C) = 69.2 ms + 19.50 µs/tok·C + 0.205 ps/tok²·C²`) and read off the excess at the multi-hop
points — +21.7 ms at 4096 (h=2) and +21.3 ms at 2048 (h=4). Two different hop counts costing the
same looked like a clean "flat multi-hop adder". It was mostly **extrapolation error**: the fit was
built from 8192/16384/32768 and then evaluated at 4096 and 2048, below its range.

The direct measurement holds the chunk fixed and shrinks the sliding window, which changes the hop
count and nothing else (`GEMMA4_SWA_WINDOW_OVERRIDE`, added to
`tt/attention/__init__.py` as a diagnostic and reverted afterwards):

| chunk | window | hops | chunk-0 |
|---:|---:|---:|---:|
| 4096 | 1024 | **2** | 174.2 ms |
| 4096 | 512 | **1** | **165.3 ms** |
| 8192 | 1024 | 1 | 242.7 ms |
| 8192 | 512 | 1 | **232.9 ms** |

The 8192 pair is the control — hop count is 1 either way, so its 9.8 ms gap is purely the halved
sliding-layer attention work. Scaling by the op's work-unit/depth structure (4096: 64 units at
depth 1; 8192: 128 units at depth 2) puts the compute saving at 4096 at ~4.4 ms, leaving **~4.5 ms
of actual multi-hop protocol cost**, ~0.09 ms per sliding layer. Upper bound, if none of the 8.9 ms
is compute: 8.9 ms.

**Consequence: there is no lever here.** Multi-hop overhead, extra fabric links, a fabric mux, and a
line-multicast halo (slot indexed by `sender_ring_id mod h`) are all worth ~nothing — the halo
payload is invariant at ~2 MiB/device/layer and the protocol costs a few ms. Chunk 2048's 131.3 ms
is close to the floor for that chunk size.

**Corrections to earlier sessions' claims, from this data:**
* "K grows ~13.4 ms per hop" — **wrong**, an artifact of fitting two points that differed in BOTH
  chunk size and hop count.
* "the multi-hop adder is a flat 21.5 ms" — **also wrong**; ~13 ms of that was extrapolating the
  quadratic fit below its range and ~4 ms was reduced attention compute in the diagnostic.
* "slope ∝ C², so the prefix term is chunk-invariant" — only true at the top end. Measured
  1.458 → 2.995 → 11.990 is 2.05x then 4.00x. Use measured per-C values, not a parametric form.
* The fixed floor is ~69 ms at the one-hop end, not the 105 ms a 2-point linear fit suggested.

**Method note:** every parametric fit in this investigation that was extrapolated outside its fitted
range produced a wrong conclusion, twice in a row. The per-chunk measured values are cheap (one
9m53s sweep gives all five); use them.

## 11.4 Deploying a single chunk size

Ratio to the best setting at each ISL:

| chunk | 1k | 4k | 8k | 32k | 128k | 256k | worst |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 1.00x | 1.52x | 2.20x | 2.46x | 2.54x | 2.61x | 2.61x |
| **4096** | 1.33x | 1.00x | 1.45x | 1.60x | 1.58x | 1.57x | **1.61x** |
| 8192 (today) | 1.85x | 1.39x | 1.00x | 1.13x | 1.19x | 1.25x | 1.85x |
| 16384 | 3.38x | 2.55x | 1.83x | 1.00x | 1.03x | 1.05x | 3.38x |
| 32768 | 7.07x | 5.33x | 3.82x | 1.00x | 1.00x | 1.00x | 7.07x |

**4096 has the lowest worst case of any single setting (1.61x vs 8192's 1.85x)**: 39% better TTFT
below 8k tokens for 25% at long context, crossover exactly at ISL = chunk. 2048 is a hard pass
(2.09x worse overall). Today's 8192 is also not the throughput optimum — 32768 is 20% faster at 256k.

## 11.5 Trap that cost ~40 minutes

`ninja -C build_Release ttnncpp` reports success and links `build_Release/ttnn/_ttnncpp.so`, but
`ttnn/ttnn/_ttnn.so` has its DT_NEEDED on `build/lib/_ttnncpp.so` and **nothing copies between
them**. Three device results were produced against the previous binary — two tests "passing" on old
code and a third failing on a `TT_FATAL` already deleted from the source. `build_metal.sh` uses
`target="install"`; the only command that refreshes what the tests import is

    cmake --build build_Release --target install

Kernel `.cpp` edits ARE JIT-compiled and do take effect, which is what makes the mixed case so
confusing. Always check `ls -la build_Release/lib/_ttnncpp.so` before trusting a device run.

---

# 12. How to reproduce the 2k and 4k chunk runs

## 12.1 Environment

```bash
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME
export HF_HOME=/path/to/hf_cache
export HF_HUB_OFFLINE=1
export HF_MODEL=google/gemma-4-31B-it
# Weight cache MUST have been built for the mesh you are about to run. A cache built for a
# different mesh shape deadlocks in layer 0 rather than erroring -- check provenance first.
export TT_CACHE_PATH=$HF_HOME/tt_cache/google--gemma-4-31B-it
export PYTEST_TIMEOUT=7200
```

## 12.2 Build

```bash
cmake --build build_Release --target install   # NOT `ninja <target>`: see 11.5
ls -la build_Release/lib/_ttnncpp.so           # mtime must be newer than your last edit
```

## 12.3 The runs

Chunk 0's device time is the TTFT for any prompt that fits in one chunk, so a `ctx_32k` run is all
that is needed for the TTFT number; the remaining chunks give the per-prefix slope.

```bash
cd $TT_METAL_HOME

# 4k chunk on 8x4 (CP8/TP4) -- 2 halo hops, one per fabric link
timeout -k 10 900 ./python_env/bin/python3 -m pytest \
  models/demos/gemma4_d_p/demo/text_demo_prefill.py \
  -k "readback_final-ctx_32k-chunk4096-text-8x4" -sv

# 2k chunk on 8x4 -- 4 halo hops time-sharing 2 links (needs this branch)
timeout -k 10 900 ./python_env/bin/python3 -m pytest \
  models/demos/gemma4_d_p/demo/text_demo_prefill.py \
  -k "readback_final-ctx_32k-chunk2048-text-8x4" -sv

# all five chunk sizes in one device session (~10 min), the table in 11.2
timeout -k 10 2400 ./python_env/bin/python3 -m pytest \
  models/demos/gemma4_d_p/demo/text_demo_prefill.py \
  -k "readback_final and ctx_32k and 8x4 and (chunk2048 or chunk4096 or chunk8192 or chunk16384 or chunk32768)" -sv

# 2k on 4x8 (CP4/TP8) needs only 2 hops, so it also runs without link sharing
timeout -k 10 900 ./python_env/bin/python3 -m pytest \
  models/demos/gemma4_d_p/demo/text_demo_prefill.py \
  -k "readback_final-ctx_32k-chunk2048-text-4x8" -sv
```

Expected chunk-0 lines (8x4), which is what to diff against:

```
[traced_perf] chunk 1/16 [0, 2048) device=131.2ms       # 4 hops, link-shared
[traced_perf] chunk 1/8  [0, 4096) device=174.0ms       # 2 hops
[traced_perf] chunk 1/4  [0, 8192) device=242.8ms       # 1 hop, unchanged baseline
```

## 12.4 Correctness gate

```bash
timeout -k 10 900 ./python_env/bin/python3 -m pytest \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py \
  -k "chunked_sliding_linear_topology_accuracy or multi_hop_sliding_halo_linear_topology_accuracy" -sv
```

Runs in ~70 s on a warm kernel cache and must report `3 passed`. These are SP8 + linear fabric, the
layout the model actually runs: 1 hop, 2 hops (one per link) and 4 hops (two per link, chained).

## 12.5 Adding other chunk sizes

`PREFILL_CHUNK_SIZES` in `models/demos/gemma4_d_p/demo/text_demo_prefill.py` lists what is
parametrized. A chunk is legal when it divides `max_seq_len`, contains whole CP-local tiles, and
needs no more hops than the ring has ranks: `hops = ceil(window / (chunk / CP))`, so at CP8 with a
1024-token window, chunk 1024 would need 8 hops -- allowed by the op, but the halo would then span
the entire ring.

## 12.6 Isolating the hop cost again (11.3)

The window override used for that measurement is a diagnostic and is NOT in this branch. To redo it,
make `self.sliding_window_size` in `models/demos/gemma4_d_p/tt/attention/__init__.py` read an env
var, then run chunk 4096 at window 512 (drops 2 hops to 1) and chunk 8192 at window 512 as the
control (stays at 1 hop, so its delta is pure attention compute).
