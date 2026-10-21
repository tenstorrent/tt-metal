# Sparse KV push for GLM-5.3 prefill: status and handoff notes

This page is for whoever picks the work up next. It covers what the sparse KV push does and where the code
lives, what was tried and how it went, what limits speed today, and what is still open.

All numbers are measured on an 8x4 Blackhole Galaxy with `FABRIC_2D_TORUS_XY`. The KVPE cache is scaled
FP8 (656 B rows, stored as 704 B DRAM-aligned pages). Inputs are real GLM-5.3 top-k indices taken from the
1M golden trace (`glm53-1020k-last5120`, last 5120-row chunk at a 1,039,360-token prefix), unless the table
says otherwise.

## Problem

In chunked prefill, every MLA layer needs the KVPE rows of the whole prefix on every chip. Today they arrive
through `high_bw_all_gather`, a dense gather of the prefix across the full mesh:
- At a 1M prefix it moves the entire 1M-row prefix to every chip in every layer.
- That takes about 8-9 ms per layer, at about 94% of the fabric roofline, so the op itself has little
  headroom left.
- Attention (`sparse_sdpa`) needs only about 1 ms per 5k chunk, flat in prefix length.

GLM-5.3 has 78 layers:
- 21 full (indexer) layers compute top-k.
- 57 shared layers reuse the top-k indices of the full layer before them.

Each chip's top-k touches about 9k-53k distinct prefix rows (median ~30k, about 3% of a 1M prefix). A row is
wanted by about 5-6 chips on average.

## What is on this branch (`ipotkonjak/sparse-mla-plumbing`)

**Scope: shared layers only.** Full layers keep the dense gather, overlapped with top-k on sub-devices
(top-k on 80 cores ∥ gather on 40). Pushing on full layers has not been agreed on and is not part of this
work.

**Data flow, once per full layer:**
1. **Plan.** `ttnn.experimental.sparse_push_plan` turns this chip's top-k indices into two outputs:
   - a per-requester bitmap table of the rows needed from each owner, plus per-owner counts;
   - compact indices, which are the top-k rewritten to slots in a compact buffer.
   It runs on up to 120 cores, and the work is bounded by the largest referenced row.
2. **Table exchange.** `high_bw_all_gather` collects the tables, so every owner knows who wants which of
   its rows.
3. **Push.** `high_bw_all_gather` called with `push_stripe_chunk` / `push_table` dispatches to
   `HighBwSparsePushFactory`. Each owner reads the rows that are wanted and sends them as fabric unicast
   straight into each requester's compact buffer.
   - Rows are packed P=8 to a packet: the model's fabric payload is 6208 B, and pages are 704 B.
   - There is one sender per (link, direction), 8 per chip, each behind a fabric mux with 2 senders per
     channel.
   - Routes follow the fabric routing tables (see Routing).
4. `sparse_sdpa` reads the compact buffer through the compact indices. No kernel change was needed, and
   the result is bitwise equal to the dense path.

**Overlap with attention.** Two pieces work together:
- **Fused prefix push.** A layer whose successor is a shared layer runs `sparse_sdpa` with `push_*`
  arguments. Attention runs on 80 cores; the push reader, senders and mux run on the other 40 cores in
  the same program, so no sub-device manager is involved. That push carries the next layer's prefix rows,
  meaning rows written by earlier chunks.
- **Resume.** Senders record their progress as `[sent, finished, turn]`. The next layer calls
  `high_bw_all_gather` with `push_part=3`, which resumes from that progress and pushes the current chunk's
  own rows.
- Compact buffers alternate by layer parity, so layer L can read its buffer while L+1's buffer fills.

**Files:**
- Model: `models/demos/deepseek_v3_d_p/tt/mla/sparse_kv_push.py` (planner cache, buffers, `plan`,
  `plan_in_region`) and `tt/mla/mla.py` (`_sparse_push_plan*`, `_next_layer_prefetch`, the push and resume
  call sites).
- Plan op: `ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/sparse_push_plan_device_operation.*`
  and `kernels/sparse_push_plan.cpp`. Its NCRISC helper does half of the OR reduce and half of the rewrite.
- Push: `high_bw_sparse_push_factory.cpp`; `kernels/sparse_push_table_reader.cpp`, `sparse_push_reader.cpp`,
  `sparse_push_writer.cpp`; and `sparse_push_route.hpp` (first-hop / tie-split routing, shared by the
  reader and writer).
- Fused attention + push: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sparse_sdpa_fused_push_factory.cpp`.

**Env knobs:**

| Knob | Default | Meaning |
|---|---|---|
| `TT_SPARSE_KV_PUSH` | 0 | Enables the push on shared layers. |
| `TT_SPARSE_KV_PUSH_OVERLAP` | 1 | Fuses the next layer's prefix push into `sparse_sdpa`. Set 0 to push serially. |
| `TT_SPARSE_KV_PUSH_PLAN` | device | `host` selects the old host planner (~190 ms; reference only). |
| `TT_SPARSE_PUSH_PLAN_OVERLAP` | 0 | Plans inside the top-k ∥ gather region (see "Plan-op hiding"). |
| `SPARSE_PUSH_TIE_SPLIT` | on | Splits ring-distance ties between both directions. Set 0 for the routing-table default. |
| `SPARSE_PUSH_MUX_K` | 2 | Senders per fabric-mux channel. 0 means 8 direct senders. |
| `SPARSE_PUSH_FUSED_QUIT` | off | Stops the fused push when attention finishes; the resume then pushes the rest. |
| `SPARSE_PUSH_PLAN_TIMING` | off | Writes per-phase timestamps into the plan table header. |
| `SPARSE_PUSH_SKIP` | - | `reads` or `sends`: times one side of the push without the other (profiling). |
| `SPARSE_PUSH_MUX_BUFFERS` | - | Number of mux buffers (no effect measured). |

## Results

**End-to-end, per-chunk forward time, all 78 layers (untraced).** The published numbers come from branch
`ipotkonjak/sparse-push-e2e-measured` (`a1819724f8c`). That version used the device plan, routed push
without tie split, and a prefetch running on its own sub-device rather than fused into `sparse_sdpa`.

| Chunk | Prefix | Gather | Push | Push + prefetch | Push × | Push + prefetch × |
|---|---|---|---|---|---|---|
| 0-9 | 0-46k | 0.50-0.54 s | ≈ gather | +10-17 ms | 0.98-1.00 | 0.96-0.98 |
| 20 | 102k | 0.581 | 0.598 | 0.567 | 0.971 | 1.025 |
| 50 | 256k | 0.670 | 0.624 | 0.591 | 1.074 | 1.134 |
| 100 | 512k | 0.850 | 0.718 | 0.683 | 1.183 | 1.244 |
| 150 | 768k | 1.013 | 0.771 | 0.735 | 1.315 | 1.380 |
| 203 | 1,039k | 1.192 | 0.823 | 0.789 | 1.448 | 1.511 |

This branch (`e531b2443a9`: fused push, tie split, plan after the region), same sweep, same paths:

| Chunk | Prefix | Gather | Push | Push + prefetch | Push × | Push + prefetch × |
|---|---|---|---|---|---|---|
| 0-9 | 0-46k | 0.52-0.55 s | ≈ gather | +0-18 ms | 0.98-1.04 | 0.97-1.00 |
| 10 | 51k | 0.533 | 0.546 | 0.543 | 0.976 | 0.982 |
| 15 | 77k | 0.552 | 0.557 | 0.553 | 0.990 | 0.997 |
| 20 | 102k | 0.581 | 0.591 | 0.563 | 0.983 | 1.031 |
| 25 | 128k | 0.577 | 0.563 | 0.557 | 1.025 | 1.036 |
| 30 | 154k | 0.586 | 0.564 | 0.547 | 1.039 | 1.071 |
| 40 | 205k | 0.616 | 0.582 | 0.555 | 1.059 | 1.109 |
| 50 | 256k | 0.670 | 0.618 | 0.597 | 1.084 | 1.122 |
| 60 | 307k | 0.720 | 0.661 | 0.627 | 1.089 | 1.148 |
| 80 | 410k | 0.764 | 0.655 | 0.621 | 1.167 | 1.230 |
| 100 | 512k | 0.850 | 0.707 | 0.675 | 1.202 | 1.258 |
| 125 | 640k | 0.948 | 0.745 | 0.712 | 1.272 | 1.331 |
| 150 | 768k | 1.013 | 0.763 | 0.728 | 1.328 | 1.391 |
| 175 | 896k | 1.098 | 0.772 | 0.741 | 1.423 | 1.480 |
| 203 | 1,039k | 1.192 | 0.816 | 0.782 | 1.460 | 1.524 |

Against the measured branch, push+prefetch is 6-9 ms faster per chunk from 256k up (1.511× → 1.524× at 1M).
The break-even prefix is unchanged at about 100k.

**Per shared layer, op level, at 1M** (`sparse_sdpa` + push of the next layer's prefix + resume, exact):

| Layer | Fused, today | Fused + tie split | Dense gather + sdpa |
|---|---|---|---|
| L14 | 1024 µs | 1040 µs | ~8.9 ms |
| L58 | 1097 µs | 1062 µs | ~8.9 ms |
| L62 | 1946 µs | 1704 µs | ~8.9 ms |

## What was tried

### Kept

| Change | Effect | Commits |
|---|---|---|
| Push of only the needed rows into a compact buffer, with indices rewritten | Exact; at 1M a shared layer drops from 8.9 ms to 1.3-2.2 ms | 603c92ce332, 51d1fb6835a |
| One sender per (link, direction), 8 per chip, chosen by first-hop direction | 32k rows/chip: 2417 → 1543 µs | 53b0a1a528b |
| Fabric mux with K=2 senders per channel | About 10%; saturates at K=2 | 7b871854f43 |
| On-device plan op (replaces the ~190 ms host plan + sync) | 1.37 ms, then 1.06 ms, then 0.81-0.90 ms at 1M (segments, NOC zero fill, register OR, extent-bounded dirty-chunk reduce, NCRISC sharing work) | e0ab914d668..d0a34a46db7 |
| Prefix / remainder push parts, and prefetch of the next shared layer during `sparse_sdpa` | Hides the prefix push behind attention | c8e26979062, f89b7347d45 |
| Push fused into `sparse_sdpa` (no sub-device manager) with progress-based resume | Matches the sub-device version (1.29/1.38/2.24 ms) and needs no sub-device switching | d1f83fe0f47 |
| Tie split on by default | L62 1946 → 1704 µs; about 1.2× on median and heavy layers' push | e128867af76, 2bbb58a5325 |

### Not kept

**Dense resumable prefetch** (`ipotkonjak/hbag-resumable-prefetch`). The next layer's dense gather started
during `sparse_sdpa` on spare cores and resumed from a device-side progress tensor.
- It gives -19% at 5k/205k (fp8).
- At 1M a 5k-chunk attention window hides only about 8% of the gather.
- The sparse push beats it above about 250k prefix; dense overlap is better below about 200k.
- The op support it added stays: a resumable `high_bw_all_gather` window and a `sparse_sdpa` core subset.

**`dispatch_fabric2d` as the transport** (4c84bec5e0b). It reaches only about 3.4 GB/s per chip at
704 B rows (one row per packet), against about 18 GB/s for the push.

**More mux buffers, more senders, or larger K.** No gain. The push isn't limited by injection slots (see
Bottlenecks).

**Changing the packet size.** The fabric payload is set once for the whole model (6208 B, from GLM53Config
`FABRIC_PAYLOAD_SIZE` 6144 + 64 combine routing bytes), so it isn't a free knob.

**One-hop unicast relay** (`ipotkonjak/sparse-push-relay`). Each row travels one hop per program round and
is staged on intermediate chips.
- The first run hung. A CB-ring wrap bug was then found and fixed, and the plan was verified on the host,
  but it was never re-run on the device.
- With tie split, the routed push already uses the same paths a unicast relay would, so a relay can only
  add staging cost.

**Plan-op hiding** (`TT_SPARSE_PUSH_PLAN_OVERLAP=1`, off by default). This runs the plan op for the first
shared layer inside the full layer's top-k ∥ gather region.
- *Plan on the top-k cores, queued before the gather:* the region went from 9.1 to 17 ms, because the plan
  waits for top-k and the gather queues behind it.
- *Queued after the gather:* the region is 9.75 ms. Top-k (~9.1 ms) is the longer leg of the region, not
  the gather (~8.0 ms), so the plan still lands after top-k. The chunk time is 1.104 s vs 1.084 s for
  planning after the region, which is within noise.
- *Plan on the gather cores (where the ~1 ms of slack is):* hung at 1M. Not investigated; a plausible
  conflict is with the gather's fabric/mux cores or its global semaphores.

### Tried, hit a wall: multicast-tree / dimension-ordered push (`ipotkonjak/sparse-push-tree`)

**Idea.** Rows are wanted by about 5-6 chips each, yet unicast sends one copy per requester, all over the
same N/S links. A multicast tree instead sends each row once per link of a tree from its owner to all its
requesters, relayed one hop at a time.
- First version: host-planned trees, Y-first or X-first chosen per group to balance links.
- Final version: dimension-ordered paths. A row goes along the owner's column to each target row, then
  along that row, each leg the short way round, with ties split by owner (row + column) parity. Because
  the paths are fixed per owner, a device planner (`ttnn.experimental.sparse_push_tree_plan`) can classify
  rows 32 at a time with bitmap ORs.

**Push results.** Bitwise exact, about 2× faster than the routed push:

| Layer | Routed | Tree |
|---|---|---|
| L14 | 388 µs | 170 µs |
| L58 | 882 µs | 403 µs |
| L62 | 1769 µs | 863 µs |

**The wall is planning.**
- Device planner speed: 21 ms, then 6, 3.4, 1.8, and finally 0.55-0.87 ms.
- With the plan op (table only, `rewrite=False`) and the table gather included, planning per full layer is
  1.16-1.53 ms, against 0.78-0.86 ms today.
- The planning increase eats most of the push saving: about 1-2% e2e at 1M.
- It comes back if the tree planner gets close to today's planning cost, for example by merging it into
  `sparse_push_plan` so there is one pass over the bitmaps.

## Bottlenecks (why the push is as fast as it is)

**Routing.** The 2D route lives in the packet header. Only Y then X is allowed; the router forbids
X-first, and the sender must use the connection of the first hop. "Leaving routing to fabric" is what the
push already does.
- With Y first, 28 of 31 destinations take their first hop on N/S, so N/S links carry about 2× the E/W
  load.
- The routing tables send every ring-distance tie N or E. Tie split sends half of the ties the other way,
  which is a free ~1.2×.

**Link-bound, not injection-bound.**
- The push runs at the speed of the busiest link direction: about 24 GB/s over its 2 links, with N links
  saturating at about 12 GB/s each because of pass-through back-pressure.
- Senders are blocked on fabric slots 71-76% of the time.
- A link on its own reaches about 20 GB/s even over multiple hops.
- That is why more senders, more mux buffers or a higher K don't help, while fewer bytes on the N/S links
  (tie split, tree) do.

**Heavy layers.** On heavy shared-layer groups the push outlasts attention.
- A host model puts this at 14 of 21 groups routed and 11 of 21 with tie split.
- That is about 15 ms per 1M chunk routed and about 7 ms with tie split.
- This is now the biggest remaining overhead per shared layer.

**Fixed costs.** These are why the push loses below about 100k prefix:
- Planning: 0.78-0.86 ms per full layer (plan op about 0.73 ms on 120 cores, plus about 0.06 ms for the
  table gather), about 17 ms per chunk over 21 full layers.
- Resume: 80-110 µs per shared layer, of which 55 µs is the cost of a push with nothing to send.

**Plan op.** It is bound by RISC-V L1 loads (8-12 cycles each) in the OR reduce and the index rewrite.
Going from 80 to 120 cores barely helps (758 → 733 µs).

## Open work, in rough priority

1. **Shallow-prefix policy.** Use the dense gather below about 80-100k prefix, where the push paths lose
   2-4%.
2. **Heavy layers.** Reduce N/S bytes (a cheaper tree planner, see above), or start the prefix push
   earlier.
3. **Plan-op speed.** About 17 ms per chunk at 1M. Options:
   - fewer L1 passes;
   - planning once per row block instead of once per requester;
   - hiding the plan on the gather sub-device by debugging the hang above.
4. **Resume fixed cost.** Skip the resume program when a layer has nothing left to send (55 µs).
5. ~~**Trace safety.**~~ Done (2026-10-20), see "Trace safety and the padded-chunk fix" below.
6. **Upstreaming.** Fold the prototype env knobs into op arguments or remove them, and split into PRs:
   - op support (resumable window, `sparse_sdpa` core subset);
   - the plan and push ops with the fused `sparse_sdpa`;
   - the model wiring.

## Trace safety and the padded-chunk fix (2026-10-20)

**Trace safety.** On the metadata path (runner trace, `PREFILL_USE_TRACE=1`) nothing in the push depends on a host
value that changes per chunk or per request:
- `_sparse_pushed_attn` passes `metadata` to `_prepare_sparse_kv_prefix`, so the KV write lands at the live chunk start.
- The plan op reads the chunk start from `metadata[1]` (`chunk_start_tensor`); it is no longer part of the planner cache
  key (every full layer `invalidate()`s), and the per-value `chunk_start()` `from_torch` cache is not used.
- The standalone push takes the slot from `metadata[0]` (`input_batch_index_tensor`, recomposed on device as
  `user * layer_num + layer`). The fused `sparse_sdpa` push gained the same: `push_batch_index_tensor`,
  `push_batch_slot_num_layers`, `push_batch_slot_layer_idx` (runtime args, re-patched on cache hits).
- Planner buffers, semaphores and progress are created by the warm pass, before capture. The host planner and
  `TT_SPARSE_PUSH_PLAN_OVERLAP` stay eager-only (asserted).

`test_sparse_mla_kv_push_traced_metadata_slots` captures a full -> shared -> shared chain once and replays it for 7
interleaved (user slot, chunk start) pairs; every replay is bitwise equal to the eager dense-gather chain. Freezing the
push slot on the host (negative control) fails it with PCC 0.0.

**Padded-chunk bug (fixed).** `sparse_push_table_reader.cpp` took the prefix/remainder split as block row
`header[4] / N` for every owner. That holds only when the chunk start is slab aligned. A padded chunk starting mid-slab
(e.g. 43552 = block 272 = 8.5 rows of 32) gave owners below `split % N` one prefix block too few, while the plan's
per-owner prefix counts included it, so rows landed in the wrong compact slots. GLM L78 padded mid15k decoder PCC went
from 0.90 to -0.04, starting at the first shared layer of the second chunk. The fix computes the owner's own count,
`ceil((split - rank) / N)`. Only parts 1 and 3 (fused prefetch + resume) use the split, so single-layer tests never saw
it. `test_sparse_mla_kv_push_prefetch_chain_padded` fails on the old reader (PCC 0.939 / 0.973) and is bitwise with the
fix; L78 padded notrace/traced now give the same PCCs as the dense gather (0.864355 / 0.981146 / 0.901564).

**Measured** (disaggregated-prefill runner, SC1 1M, traced, 14 kW, chunk ms / TTFT at 1020k): main 1500.7 / 199.8 s;
push 835.4 / 136.9 s; push + PR #61387 fabric gather on the full layers 751.4 / 128.9 s (untraced 819.4 / 146.5 s).
Below about 100k prefix PR #61387 alone is slightly faster than push + fabric gather (TTFT -4.9% vs -3.6% at 100k).

## How to run

All device runs go through `scripts/run_safe_pytest.sh`, which takes the device lock and resets the device
after a hang.

- **Exactness**, MLA level (push vs dense gather, bitwise, single and multi-chunk, bf16 and fp8):
  - `models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py -k kv_push`
  - prefetch chain: `-k test_sparse_mla_kv_push_prefetch_chain`
- **Op level, real 1M top-k:** `tests/nightly/blackhole/sdpa/test_sparse_sdpa_gather_overlap.py`
  - `test_sparse_overlap_device_plan`
  - `test_sparse_resume_breakdown`
  - `test_sparse_plan_beside_gather`
- **Plan op vs the host reference:** `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_sparse_push_plan.py`
- **End-to-end depth sweep:**
  `models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_glm_prefill_sparse_push_depth_sweep`
  - Needs `MESH_DEVICE=TG TT_SPARSE_KV_PUSH=1 PUSH_SWEEP_N_CHUNKS=204`.
  - Needs the GLM-5.3 weights (`GLM53_HF_MODEL`), the TTNN cache (`TT_GLM53_PREFILL_TTNN_CACHE`) and the
    golden trace (`PREFILL_TRACE_DIR`).
  - `PUSH_SWEEP_AB=<chunks>` limits the A/B samples to the listed chunks.

## Kernel pitfalls hit along the way

- **L1 → L1 NOC transfers of 4 B need the same 16 B offset at the source and the destination.** A 4 B
  counter read at offset 4·s returns the word at the start of its 16 B group. A misaligned 4 B write hung
  every chip and needed a Galaxy reset.
- **The plan op's NCRISC helper must always launch.** It does half of the OR reduce, not just the rewrite.
- **Program semaphores are re-initialized on every launch**, so progress that has to survive between
  programs goes through DRAM (the progress tensor).
- **DPRINT chip ids are physical device ids, not mesh ranks.**
- **Profiling is unreliable here.** On this build, Tracy and the realtime profiler give wrong or missing
  numbers for sub-device programs. Time with host `perf_counter` around synchronized replays, or with
  `SPARSE_PUSH_PLAN_TIMING`.
