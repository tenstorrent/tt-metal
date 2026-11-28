# GLM-5.3: move the shared-layer KV push to sparse_kv_dispatch

Status: plan v3 (after review rounds 1 and 2: Codex gpt-6.2-astra, Claude Code CLI; log in section 11).
Branch `pjosipovic/kv-dispatch-glm`: the K3.1 dispatch experiment rebased on main `4212b8a95a8`.
Depends on #67278 (real-data perf test, `LayerOutputCapture`, the 517K real-prompt capture): this work rebases on it
once it merges; until then Phase 0 runs on a branch with #67278 merged in locally.

## 1. Goal

Make the KV rows GLM-5.3's shared layers need arrive as fast as possible, on every Galaxy mesh and fabric (8x4 and
4x4; LoudBox and QuietBox are out of scope for now and keep today's transport), with one
transport: a productized `sparse_kv_dispatch` (store-and-forward trees; `KVD_DESIGN=3` in the experiment).
Success is judged by **shared-layer time and TTFT**, measured as section 6 defines.

**In scope:** every row movement for shared (indexer-reuse) layers:
- the full layer's PREFIX pre-send for the next shared layer, beside the full layer's sparse SDPA (`mla.py: _presend`);
- a shared layer's own rows: ALL, or REST after a pre-send (`_sparse_pushed_attn`);
- a shared layer's PREFIX pre-send for the next shared layer, or the next chunk in packed forwards (`PushStep`).

**Out of scope:** the full layer's own prefix gather beside top-k (stays `high_bw_all_gather`); two-level top-k; the
top-k LLK rework; K3.1. K3.1 has no shared layers; its experimental full-layer dispatch (global landing, `active_from`)
exists only on the experiment branch. Decision: the productized op is compact-landing only; a K3.1 integration, if
pursued later, is redone on it. Experiment `mla.py` code and env knobs go into no PR of this plan.

## 2. Configurations and what runs today

`is_sparse_kv_push_supported` needs FABRIC_2D_TORUS_XY with at least 3 chips on each axis
(`push/sparse_kv_push.cpp:48`). So the push runs on C1 and C6 only; everywhere else shared layers gather the whole
prefix every layer (`_gather_kvpe_prefix`).

| id | mesh | fabric | sp_axis | chunk | today | host | gates |
|---|---|---|---|---|---|---|---|
| C1 | Galaxy 8x4 | TORUS_XY | 0 | 5,120 | push | b07u08; CI Galaxy | all |
| C2 | Galaxy 8x4 | TORUS_Y | 0 | 5,120 | gather | b06u08 | G1–G5, G7 |
| C3 | Galaxy 8x4 | TORUS_X | 0 | 5,120 | gather | b07u08, forced fabric | G1–G5, G7 |
| C4 | Galaxy 8x4 | FABRIC_2D (open) | 0 | 5,120 | gather | b07u08, forced fabric | G1–G5, G7 |
| C5 | Galaxy 4x4 stage (`LINE, RING`) | TORUS_X | 1 | 2,560 | gather | CI Galaxy; b07u08 (`run_4x4_stage_tests.sh`) | G1–G5, G7, G8 |
| C6 | Galaxy 4x4 carve (`RING, RING` sub-torus of a Galaxy) | TORUS_XY | 1 | 2,560 | push | CI Galaxy; b07u08 (`run_subtorus_4x4_tests.sh`) | all |
| C7 | Galaxy 4x4 | FABRIC_2D (open) | 1 | 2,560 | gather | C5 host, forced fabric + descriptor | G1–G5, G7 |

Mesh, fabric, sp_axis: `perf_cases.detect_box` / `_galaxy_box`. #67509 (open) moves the GLM 4x4 tray-half stage
(C5) to sp_axis 0; the `[RING, RING]` carve (C6) keeps sp_axis 1 (#66944 pins it), so the dispatch supports both axes
and C5 is gated on whichever its model config uses when this lands. "Forced fabric" needs a test parameter that does
not exist yet (work item W1). Phase 0 cannot start on a config until its host is named.

Measured so far (C1 only, b07u08, traced, ALL part, op alone, no attention running), µs per call on #67278's real
top-k (L42 = most distinct rows per chip, L58 = median):

| case | push, 24 cores | dispatch, 40 cores (16 planners, K=2) | dispatch, full grid (48 planners) |
|---|---|---|---|
| L58 64K / 128K / 256K / 512K | 509 / 572 / 857 / 982 | 267 / 311 / 413 / 513 | 199 / 225 / 292 / 345 |
| L42 64K / 128K / 256K / 512K | 914 / 947 / 1231 / 2548 | 378 / 474 / 634 / 1544 | 279 / 310 / 413 / 762 |

In #67278's real-data test the L42 512K shared layer takes 3.50 ms traced, ~1.7 ms of it waiting on the push.

## 3. Design

### 3.1 Op `ttnn.experimental.sparse_kv_dispatch`
- **One design:** v3 only. Delete v1/v2 kernels (`kvd_planner.cpp`, `kvd_port_sender.cpp`, `kvd_port_unpacker.cpp`),
  `kvd4`, the `KVD_DESIGN` / `KVD3` / `KVD_ABL` env knobs and stats builds. Geometry (planners, K, lanes, pool and
  stage depths) becomes op arguments with validated defaults.
- **No DRAM scratch:** v3 never reads it (`kvd3_planner.cpp:161`); remove `KvDispatchScratch` (~100 MB per chip).
- **Compact landing** (3.2) in both writers: workers (remote rows) and planners (this chip's own rows,
  `kvd3_planner.cpp:537-555`).
- **Parts** `ALL | PREFIX | REST`, the push's meaning (PREFIX = rows in blocks before the requester's split block).
  Planners take each requester's split and active extent from its own plan slot and set a requester's mask bit only
  for rows in the part. An empty part still runs fences, ENDs and credits. `active_from` is removed.
- **sp_axis 0 and 1:** stripe owner o maps to physical chip `chip(o)` (identity for sp_axis 0; `column * rows + row`
  -> row-major physical index for sp_axis 1) in the planner routes, the TX slice exchange (`dest_chip_ids`) and the
  mask bits; global row ids are rebuilt with the logical owner.
- **Supported set** (`is_sparse_kv_dispatch_supported(mesh, stripe, max_rows, payload)`): 2D meshes 2x2 up to axes of
  8 and 32 chips; FABRIC_2D, TORUS_X, TORUS_Y, TORUS_XY, but a wrap only on axes of 3 or more chips (a wrapped
  2-chip axis has duplicate neighbours, `sparse_kv_dispatch_device_operation.cpp:197-207`); `max_rows` a whole number
  of slabs (`chips x stripe`); 1 or 2 links. Everything else returns False and has rejection tests.
- **Deadlock freedom:** `tools/kvd3_model.py` (on `pjosipovic/topk-push-experiment`) is ported, extended to line and
  partial-torus class depths, and run in Phase 0 and in the op's unit test.
- **core_grid** for the beside-attention sub-device; landing pools on port cores only (done).
- **Trace-safe:** chunk start from the plan table, cache slot from `batch_index_tensor`; all four semaphores zero after
  every call.

### 3.2 Compact landing
- Requester r's compact page for row g = rank of g among r's needed rows in global row order.
- Output spec: rows = min(query rows x top-k, cache rows) per chip, no per-owner padding (GLM: 327,680 rows). It always
  fits the dense gather buffer, so dispatch always reuses the dense buffer as one parity (the push's two-push-buffer
  scheme does not arise for dispatch; it stays reachable only with `sparse_kv_transport="push"`).
- Rank lookup per landed row: block base rank (4 B per block, in each writer core's L1: ~26 KB at 1M, counted in the
  L1 budget) plus a popcount over the block's `stripe / 32` bitmap words read from the plan table in DRAM, batched
  per packet.
- The plan's compact indices use the same function, so sparse SDPA reads what the dispatch wrote.

### 3.3 Plan `sparse_push_plan(owner_major=True, compact=True)`
- Always owner-major for dispatch: multi-packet slots allowed (v3 TX fragments them, `kvd3_tx.cpp:232-266`). The
  push-era one-packet condition (`sparse_kv_push.py:296-301`, `mla.py:850-857`) does not apply to dispatch plans.
- Outputs: owner-major slots, block base ranks, compact indices.
- **Exchange once per build:** each plan build gets a generation number in its table header and its own inbox; the
  first dispatch call on a build exchanges and stamps the inbox with that generation; later calls (PREFIX, REST,
  later shared layers, either grid) check the stamp and skip the exchange. A rebuild at the same address bumps the
  generation, so a stale inbox is never used.
- One plan per (indices, chunk start), cached per chunk as main does (#67059).

### 3.4 Plan once, dispatch many (Phase 0 decides)
The first dispatch per plan build writes the planners' per-port send lists to DRAM beside the table; later calls
replay them. Decision rule in Phase 0 (section 5, step 4).

### 3.5 GLM integration
- `SparseKvPushPlanner` keeps the cache, buffers, parity and pre-send bookkeeping; its op calls switch to the dispatch
  where supported. This also moves shared layers on C2–C5, C7 from gather to dispatch.
- Pre-sends run on the beside-attention sub-device (per-box geometry from Phase 0); inline ALL / REST on the full
  grid. Program variants per config: {beside, full} grid x the parts used; all built in the warm pass, and the
  post-warm program count is asserted stable.
- Switch: model-config field `sparse_kv_transport = "dispatch" | "push" | "gather"` (default: today's behaviour until
  the flip PR). No env switches in model code. The planner counts calls per transport and part; tests assert them;
  the transport is logged once per mesh.

## 4. Work items outside the op

- **W1 forced-fabric fixtures:** `perf_cases` / `test_sparse_mla_perf.py` take an explicit box (mesh, fabric, sp_axis,
  descriptor) so C2–C4, C7 run; rejected combinations skip with a reason.
- **W2 real top-k artifacts:** the 8 dumps (L42 / L58 at 61,440 / 128,000 / 256,000 / 512,000) stored next to the
  #67278 capture (`golden_traces/glm53-real-prompt-517k/topk/`) with a sha256 manifest in the test; a fixed remap of
  owners for 4x4 meshes.
- **W4 real-data layer cases on 4x4:** `real_data_params` (#67278) runs on Galaxy 8x4 only. Extend it to the 4x4
  carve (C6) and the 4x4 stage (C5): same capture and layers (L42, L58), same cache tokens (61,440 / 128,000 /
  256,000 / 512,000; a 4x4 box holds the whole context), the box's 2,560-token chunk. The capture is per-token hidden
  states, so only the parameter list and the box's chunk change. Phase 0 checks the 512K case fits in DRAM on 16 chips.
- **W3 timing harness:** the layer perf test reports the median of 20 traced replays (today it times one) and the
  fastest of 10 untraced; same harness for every config.

## 5. Phase 0: thin-integration spike (go / no-go before productizing)

Spike code on the experiment branch, not for merge:
- **Adapter:** shared layers and the full layer's pre-send call the experiment dispatch with a minimal PREFIX / REST
  filter in the planners, global landing into the dense gather buffer plus a second dense buffer for the other parity,
  shared-layer SDPA on natural indices (as the K3.1 experiment). The `mla.py:850-857` one-packet guard is lifted; the
  transport counters are checked in every run.
- sp_axis 0 only in the spike. 4x4 runs therefore use sp_axis 0 meshes and establish routes, not axis-1 exactness
  (that is G1 of the op PR).

Steps, in payoff order (configs that gather today first), every timing per section 6:
1. **Baselines:** today's shared-layer time on C5, then C2–C4, C7, C1, C6: gate cases (warm, long); real-data
   layer cases (W4) on C1, C5, C6; the remapped real top-k elsewhere.
2. **Routes and deadlock:** spike exact on every config's mesh and fabric (synthetic + remapped real top-k); class
   model passes for each.
3. **Sequence timing:** a full -> shared -> shared sequence in the model, today vs spike, on C5, C2–C4, C7, C1, C6,
   gate cases and real cases. Recorded per layer: pre-send time beside SDPA, inline REST/ALL time, plan time.
4. **Planner share** on C1 L42 512K beside attention: (planner busy time from first to last planner, per chip, median
   over chips) / (dispatch call wall time). If above 0.30, 3.4 goes into the op PR.
5. **Geometry and L1:** planners / K / cores for the beside sub-device per box with production L1 tensors live (#67309),
   at the box's largest cache, including the per-writer rank tables.

**Go** (all must hold, from step 2 and 3 numbers, section 6 statistic):
- (a) exact on every config;
- (b) every G5 row: the spike's shared-layer time on the local host, scaled to the CI host by the ratio of today's
  CI target to today's local measurement of the same case, is at or below the row's pass value;
- (c) for rows without a target yet (4x4 real cases, C2–C4, C7): the spike beats the local baseline, long cases by at
  least 10%; those measurements set their targets.
Otherwise the numbers go to the user before any PR. At go, the spike's C1 per-part times become the G3 targets
(+10%), written into section 7 and frozen; changing them later needs the user's sign-off.

### 5.1 Phase 0 log (b07u08, branch `pjosipovic/kv-dispatch-glm`, spike commits 99ca2be3974, c2f9172aacf)

- **Op exactness, 8x4, every fabric:** `test_dispatch_exact` (fp8, small and dense picks) and `test_dispatch_parts`
  (PREFIX, REST, PREFIX + REST, ALL at splits 0 / mid / last) pass on TORUS_XY, TORUS_X, TORUS_Y and FABRIC_2D. The
  non-XY fabrics first overflowed L1 (more classes on lines: the staging and landing CBs scale with them); fixed by
  an L1 fit in the program factory.
- **L1 fit order matters:** staging depth 1 makes the dispatch ~6x slower (gate top-k: 145 -> 903 us); pool depth 2
  costs ~4%, 8 rows a packet ~20% on the full grid (and is faster beside attention). In the model, live L1 tensors
  leave ~280 KB less L1 than the bench, so the fit drops pool depth first, then rows a packet, staging last.
- **Semaphores per program variant:** sharing the four persistent words between the full-grid (inline) and the
  beside-attention (pre-send) programs let one chip's next call release a neighbour's TX still in the previous call
  (different cores in the two programs): non-deterministic rows on most chips. One set per variant fixes it; the
  production op must own its words per program (section 3.1).
- **Model exactness:** with both fixes, shared-layer outputs are bit-exact with the push on all 32 chips, and
  identical across runs (gate warm and long, 2 runs each).
- **8x4 gate cases (few distinct rows), traced ms, push / dispatch:** warm 1.968 / 1.873; long 1.992 / 2.074
  (2 runs). Untraced 0.1–0.2 ms slower with the dispatch (host side; open).
- **Isolated op on the gate top-k (µs):** push 169–213, dispatch 145–173 (faster); the in-model long-case gap is not
  the op's own time (open).

- **8x4 real-data cases (#67278), one pair, traced ms push / dispatch (ratio):** L42 512K 3.461 / 2.459 (0.71);
  L42 256K 2.110 / 1.883 (0.89); L42 128K 1.843 / 1.826 (0.99); L42 64K 1.794 / 1.854 (1.03); L58 512K 1.882 / 1.857
  (0.99); L58 256K 1.791 / 1.789 (1.00); L58 128K 1.752 / 1.777 (1.01); L58 64K 1.728 / 1.768 (1.02). All bit-exact.
  Untraced ~0.2 ms slower except L42 256K / 512K (host enqueue of 32 per-chip programs). The 64K–128K misses are the
  dispatch's fixed per-call cost where the push is already hidden: the target of "exchange once per build" (3.3) and
  of per-chip data in runtime args.

- **Every Galaxy config, gate cases, one pair, traced ms today / spike** (4x4 runs with sp_axis forced to 0 in both
  arms; the spike is axis-0 only): 4x4 stage (gather today) warm 1.956 / 1.832, long 4.311 / 1.935; 4x4 carve (push)
  warm 1.856 / 1.797, long 1.877 / 1.929 (+2.8%); 8x4 torus Y (gather) warm 2.421 / 2.175, long 5.875 / 2.323; torus
  X (gather) warm 2.140 / 1.921, long 5.578 / 2.167; open mesh (gather) warm 2.823 / 2.231, long 9.805 / 2.352.
- **Exactness off the push configs:** the spike is bit-exact with the push on 8x4 torus XY and the carve. Against the
  gather it differs on every chip with PCC ~0.99985, and so does the push itself (8x4 torus XY, push vs gather:
  PCC 0.99984): the two existing transports are not bit-identical. G4 is restated accordingly (section 7).

- **4x4 real-data cases (W4, sp_axis forced to 0), one pair, traced ms today / spike:** carve (push) L58 64K–512K
  1.667 / 1.694, 1.655 / 1.695, 1.699 / 1.777, 1.726 / 1.811; L42 1.686 / 1.766, 1.690 / 1.769, 1.785 / 1.886,
  2.660 / 2.356 (bit-exact). Stage (gather) L58 1.831 / 1.710, 2.156 / 1.710, 2.780 / 1.786, 4.060 / 1.825; L42
  1.813 / 1.741, 2.141 / 1.765, 2.782 / 1.869, 4.068 / 2.604 (PCC ~0.99987, as push vs gather). The carve loses 2–6%
  wherever its push is hidden: fixed per-call cost, next step.

- **One program per mesh (user, 2026-11-27):** the per-chip descriptors merge into one program with per-device
  runtime args (`merge_per_device_program_descriptors`), placement as `high_bw_all_gather`: each chip's port cores
  next to its own Ethernet cores, each role's kernel on the union over the chips, roles on disjoint cores. Untraced
  host cost gone (8x4 gate warm untraced 2.358 vs push 2.393; per-chip programs were 2.46–2.51). Full grid: unions
  TX 15 / workers 27 cores, same speed as per-chip programs (L42 512K 1022 vs 1024 us). The 40-core beside grid cannot
  hold the unions with 16 planners: uniform placement there, ~13% slower than per-chip programs (1733 vs 1539 us);
  K = 1 (fits per-chip) is worse (1984), fewer planners too.
- **Where the hidden-push cases lose (carve gate long, synced sections, us):** inline REST 152 push / 269 dispatch,
  attention + pre-send 1351 / 1404. A REST dealt every owned unit of the cache; it now deals only the groups from the
  split on (PREFIX those before it).

- **After PREFIX/REST deal only their units (a7f83b04d0a):** 8x4 real 64K / 128K, traced push / dispatch: L58
  1.736 / 1.743, 1.775 / 1.728; L42 1.798 / 1.789, 1.818 / 1.798 (all bit-exact). Gate: 8x4 warm 1.957 / 1.856, long
  1.998 / 1.973; carve warm 1.843 / 1.822, long 1.874 / 1.898. Untraced long still +3% (8x4) / +5.6% (carve): per
  call the dispatch enqueue costs ~30 us more host time (env lookups and hashing in the spike) and its REST ~50 us more
  device time in the model than in the bench (where its REST is faster than the push's: 102–105 vs 104–133 us).

- **Compact landing (dc06c39292b):** `sparse_push_plan(rank_table=True)` writes a per-block page (each word's rank
  within its owner's run, and the words) plus the run bases; `sparse_kv_dispatch(rank_table=...)` lands rows (workers
  and the planners' local copies) at `compact_page(run_base[owner], rank)`, the push's compact layout, so sparse SDPA
  reads them through the plan's compact indices. `test_dispatch_compact`: every compact index of every chip reads its
  row, ALL and PREFIX + REST, all four 2D fabrics. No second full-size buffer.
- **Program per mesh vs placement (user, 2026-11-27):** one program per mesh with each chip's own placement where
  the role unions fit the grid (full grid); per-chip programs, each with its own placement, where they do not (the
  40-core beside-attention grid). Not a combined TX + worker kernel.
- **Plan once, dispatch many (85fcbe878f1 + review fixes 7cf3b7db8ac, a2747d9c476):** `replay=1` records each
  planner's routing (prefix / rest streams, 16 B entries) into a DRAM record on the first dispatch of a plan build;
  later calls (`skip_exchange`) `replay=2` stream it instead of routing. Any mismatch (magic, landing, lanes, ports,
  overflow, bad user) falls back to routing. Record: ~5 MB per plan slot per chip at 512K worst case. Bench 8x4:
  -4 to -20% per call (gate 512K 197 -> 155 us). Exact on all fabrics (parts 56/56, compact 32/32).
  Safety (e8464cf78be, 1e6c58f788b): each record carries its plan build's generation (`replay_generation`, a common
  runtime arg), a replay of another generation routes instead (exact); each replaying planner writes whether it
  replayed or routed, and the tests check it after every replaying call (replay, mixed planner counts, record 48 /
  replay 4 on the GLM grid, unwritten record, stale generation incl. a program-cache hit).
- **Common runtime args:** the planners' and the op's core coordinates (the same on every chip) moved from per-core
  to common args (planner, TX, reader, helper): ~8.5k -> ~3.3k per-core words per chip and launch.
- **Carve losses were placement, not the op (fast in-process server, 2026-11-27).** Op alone (traced bench, carve)
  dispatch beats push at every case (L58 64K 177 vs 278 us; L42 512K 1076 vs 1823). In the model the losses came from:
  (1) the rest on the whole chip with 48 planners: ~70 us more launch latency untraced and ~20 us more host enqueue
  than on the 32-40-core pre-send grid; (2) 16 pre-send planners (DRAM readers) beside SDPA: attention +1-4%; the
  push itself on the dispatch's 40-core grid loses 1-3%. Ethernet proximity is not it: the pre-send's ports sit 9-16
  NOC hops from half their Ethernet cores on the 4-column grid, but a top-rows grid (1-hop ports) is no better. Not
  RISC bandwidth either: K=1 with WSPLIT=2 (NCRISC workers) loses 15% at L42 256K / 512K. Fix (defaults now): the
  whole dispatch (pre-send and rest) on the bottom 8 rows of the 4 columns beside attention (32 cores), 4 planners.
  Server, 3 interleaved reps, traced ratio vs push: L42 128K 1.001, 256K 1.022, 512K 0.879; L58 512K 1.005.

### 5.2 Phase 0 status (2026-11-28 at 6fc93d2b0a6, b07u08, 3 interleaved pairs: preliminary, section 6 asks 5; replay on, CI env; ratio dispatch / push)

Config: pre-send and the replayed rest on the bottom 8 rows of the 4 columns beside attention (32 cores), 4
planners; the rest one program (`single_program`), the pre-send per-chip placement; attention enqueued before the
pre-send; ALL (no pre-send) on the whole chip with 48 planners. Every output bit-exact vs the push (8x4 32 chips,
carve 16).

| go row | result |
|---|---|
| (a) exact on every config | yes: parts / compact / exact on all 8x4 fabrics, the carve; replay status checked per call (replay, mixed counts, record 48 / replay 4, unwritten, stale generation) |
| (b) 8x4 L42 512K at most 0.80 / L42 256K at most 0.95 | 0.734 / 0.861 traced (untraced 0.751 / 0.895) |
| (b) 8x4 other real cases at most 1.01 | yes: 0.944-0.983 traced, 0.941-0.962 untraced |
| (b) carve L42 512K at most 0.85 | no: 0.881 traced, 0.912 untraced (5 pairs, 472ae6a03da; 6 planners: 0.78 on the server, but L42 256K +4%) |
| (b) carve other real cases at most 1.01 | all but L42 256K: 5 pairs at 472ae6a03da, 1.023 traced / 1.021 untraced (a loss); others 0.969-0.986 traced, 0.952-0.987 untraced |
| 8x4 gate | shared warm 0.940 / long 0.963, full warm 0.982 / long 0.986 traced (untraced 0.940 / 0.960 / 0.994 / 0.984) |
| carve gate | shared warm 0.942 / long 0.970, full warm 0.979 / long 0.992 traced (untraced 0.942 / 0.958 / 0.974 / 0.995) |
| (b) 4x4 stage L42 256K / 512K at most 0.95 | 0.691 / 0.742 traced (untraced 0.725 / 0.772); every real case 0.455-0.928 (2 pairs, fedfe3a822e, vs the gather) |
| (c) gather configs, long at least 10% faster | shared layer long: stage 0.469, torus X 0.362, torus Y 0.380, open mesh 0.237 traced; warm 0.76-0.90. Full layer: 3-7% slower than with the gather (it now runs ALL plus the next layer's pre-send beside its attention), net per full + shared pair still far faster (torus X long: +0.36 / -3.56 ms). Outputs vs the gather: min PCC 0.99984 |
| G5 untraced rows at most 1.02 | yes except carve L42 256K (1.022) |

Full layer on gather configs (8x4 torus X gate, traced, ms; gather / dispatch without pre-send / split only, no
pre-send (`SCRATCH_PRESEND_NOOP`) / dispatch): warm 2.57 / 2.56 / 2.64 / 2.78, long 7.52 / 7.52 / 7.68 / 7.84. Its own
ALL matches the gather; half the cost is attention on 8 columns beside the sub-device split, half the pre-send's
contention. The pre-send pays on real data: stage shared layer without it is 0.10-0.52 ms slower (L58 64K 1.93 ->
2.03, L42 512K 3.22 -> 3.73); in the random-weight gate (2-7K rows a chip) the shared layer is as fast without it.

Open: the full layer on gather configs (+3-7% vs the gather: split + pre-send, above); carve L42 256K (+1-2%): our pre-send moves its rows faster and more intensely beside SDPA (attention +335 us
with it vs the push's +230 us; at 128K both ~+150); planner count / rows trade it against 512K. Carve L42 512K 0.88
(target 0.85). A silent fallback from replay to routing with 4
planners costs +18% to +58% at 256K / 512K: the planner status line and the plan generation make it testable.

Known limit (review): v3 validation sizes the grid for every mesh direction (the uniform placement), so a tiny grid
on an open 2x2 mesh that would fit per-chip programs is rejected; not a Galaxy configuration.

## 6. Measurement rules (Phase 0 and every gate)

- Env: `TT_METAL_SHM_TRACKING_DISABLED=1 LOGURU_LEVEL=ERROR`, Inspector default, CPU clock held as the perf tests do
  (`hold_full_cpu_clock`).
- **Ratios (A/B):** baseline and candidate on one host and build (switch flipped, nothing else), interleaved A B A B,
  5 pairs, one pytest process each. Per pair: candidate / baseline of the W3 statistic. Decision: median of the 5
  pair ratios. If it lands within 0.01 of the threshold, run 5 more pairs and decide on all 10; no further reruns.
- **Absolute µs (G3 row 1):** 5 runs of the W3 statistic, median; within 2% of the target, 5 more runs, decide on 10.
- Inputs: W2 artifacts by sha256; synthetic top-k from a seeded generator per box (distinct rows per chip at 10%,
  50%, 100% of `min(query rows x top-k, cache rows)`, owner skew 1x and 4x, requester overlap low / high).
- Case lists: gate cases = `SCENARIO_CACHE_TOKENS` (warm, long) per box; real cases = the 8 W2 dumps (remapped off C1).

## 7. Gates (all binary; applicability per config in section 2)

**G1. Op exactness** (op PR), 1 and 2 links where the mesh has them, against an independent host reference:
- the reference computes sorted unique valid rows per requester, every compact index, and expected bytes; compare
  raw logical row bytes (padding excluded: transport stamps live there);
- poisoned output: every needed row correct, every other page still poison; each part alone against poison, then
  PREFIX + REST == ALL, disjoint;
- cases: empty top-k; all-chips-same; single-owner hot spot; local-only and transit-only chips; corner-to-corner on
  lines; asymmetric requesters (unequal splits and active extents, so a swapped axis or stale bitmap tail shows);
  non-slab-aligned split; sentinel and out-of-range indices; multi-packet inbox slots; generation bump at the same
  address; scaled FP8 and BF16 rows; `batch_index` and `batch_index_tensor`; real top-k (remapped);
- rejection tests for every unsupported combination of section 3.1.
Pass: zero mismatches; every rejection raises.

**G2. Trace safety, determinism, liveness** (op PR):
- one capture replayed with chunk start and slot changed through metadata: bit-exact against eager each time;
- 100 back-to-back replays bit-exact; all four semaphore words zero on every participating core after each call;
- soak 1,000 calls per config locally (100 in CI) with `TT_METAL_OPERATION_TIMEOUT_SECONDS=60`; a timeout fails.
Pass: all.

**G3. Op performance** (op PR):
- C1: PREFIX beside attention (sparse SDPA running), inline REST and inline ALL on the 8 real cases: each at or below
  the frozen target (section 5); targets: _filled in at Phase 0 go_;
- compact vs global landing, identical inputs and grids, C1 8 real cases, measured in the op PR before the global path
  is deleted: median ratio at most 1.05 per case.
Pass: every row.

**G4. Model exactness** (integration PR, switch forced to dispatch in the tests):
- shared-layer attention output bit-exact vs the push on gate and real cases, eager and traced, single-chunk and
  packed forwards, where the push runs (C1, C6); on configs that gather today (C5), whose output the push does not
  reproduce bit for bit either, per-chip PCC vs the gather at least the push's on C1 (0.99984) and the GLM accuracy
  tests below unchanged;
- packed-forward bookkeeping: exact PREFIX / REST / ALL call counts and plan-build counts, as
  `test_glm_prefill_multiuser.py` checks today; cases: full -> shared -> shared, stage entry with rewritten indices at
  the same address, no successor, empty positions, slot / start changed on replay, beside and full grids alternating
  on cache hits;
- program count stable after the warm pass; transport counters show dispatch on every case;
- unchanged and passing: `test_sparse_mla.py -k glm_5_3`, `test_sparse_mla_determinism.py -k glm_5_3`, and the
  `glm_chunked_accuracy`, `glm53_trace`, `glm_packed` jobs.
Pass: all.

**G5. Layer performance** (flip PR). Measured with Slavko's tests as CI runs them: `test_sparse_mla_perf.py`
(gate cases, every box) and `test_sparse_mla_real_data_perf` (#67278; 8x4 today, 4x4 via W4), shared (no-indexer)
layer, ms on the CI host. The flip PR lowers the targets in `_TARGETS_MS` / `_REAL_TARGETS_MS` to the new
measurements; the gate is on those new targets (CI-measured, section 6 statistic for local A/B):

| box | case | today's target, traced (untraced) | expected after | pass if new target at most |
|---|---|---|---|---|
| Galaxy 8x4 | real L42 512K | 3.34 (3.62) | ~2.4 (~2.65) | **2.60 (2.85)** |
| Galaxy 8x4 | real L42 256K | 2.04 (2.38) | ~1.80 (~2.12) | **1.90 (2.22)** |
| Galaxy 8x4 | real L58 512K | 1.84 (2.14) | ~1.78 (~2.08) | 1.84 (2.14) |
| Galaxy 8x4 | other real cases, gate warm / long | as today | as today | today's target |
| Galaxy 4x4 stage | gate long | 4.41 (4.53) | ~2.2 (~2.3) | **2.50 (2.60)** |
| Galaxy 4x4 stage | gate warm | 2.10 (2.17) | ~2.0 (~2.1) | today's target |
| Galaxy 4x4 subtorus (carve) | gate warm / long | 1.98 / 2.08 (2.14 / 2.22) | as today | today's target |
| Galaxy 4x4 carve, stage | real cases (W4) | no target yet | from Phase 0 | set from Phase 0, then frozen |
| any box | full (indexer) layer, every case | as today | as today | today's target |
| C2–C4, C7 (forced fabric, W1) | gate warm / long | no target yet (gather today) | from Phase 0 | below their Phase 0 gather baseline; long at least 10% below |

"Expected after" is an estimate. 8x4: today's exposed push minus the dispatch's measured saving on 40 cores (L42
512K: push 2.55 ms vs dispatch 1.54 ms beside attention, ~0.85 ms hidden either way), scaled to the CI host. 4x4
stage: the gather is replaced by a dispatch of the gate cases' few distinct rows (2–7K per chip), so long
should land near warm. None of it includes contention with sparse SDPA; Phase 0 measures it and the pass column is
re-confirmed (only tightened, or loosened with the user's sign-off) at Phase 0 go.
Pass: every row.

**G6. End to end** (flip PR), C1 and C6 (the 4x4 carve, its 2,560-token chunk), GLM-5.3 78 layers on the #67278 517,120-token prompt with
`test_prefill_transformer_chunked.py` (the run that produced the capture), traced:
- outputs: one capture run per arm with `LayerOutputCapture` on: the captured layers' outputs bit-exact;
- TTFT: section 6 ratio with capture off in both arms, first chunk submitted to last chunk's output synchronized:
  median ratio at most 0.995 on C1 and C6.
Pass: both, on both configs.

**G7. Memory** (integration PR). Goal: the dispatch costs no memory where the push runs today, and at most one
compact buffer where shared layers gather today.

Measurement: per chip, the allocator's DRAM and L1 allocated bytes (`ttnn.get_memory_view`, cross-checked with
`ttnn.dump_device_memory_state`) after the warm pass and after the last chunk, peak of the two, in a packed GLM run at
the box's largest cache (C1, C6: 1M; C5: 512K); one run per transport, same host and build. Δ =
dispatch − today.

| item | budget (pass if at or below) | expected |
|---|---|---|
| DRAM Δ, C1 and C6 (push today) | 0 MB | −3 MB (unpadded compact buffer) − ~4 MB per cached plan (no gathered tables) |
| DRAM Δ, configs that gather today | one compact buffer + 1 MB: `min(160 x 2048, cache rows) x row page` (GLM 8x4 / 4x4: 231 MB) | that buffer |
| DRAM scratch owned by the dispatch | 0 B | 0 (the experiment's ~100 MB at 1M is deleted) |
| Dispatch plan state per cached plan (own table, inbox, base ranks) | 256 KB | ~150 KB at 1M |
| L1 tensors added by the dispatch (inbox in DRAM, nothing else outside program CBs) | 0 B | 0 |
| L1 program CBs: every dispatch program allocates in the warm pass at the largest cache, with production L1 tensors live (q and attention output in L1, #67309) | fits on every config, headroom at least 16 KB per core role (port cores, planners), logged | to be measured in Phase 0 step 5 |
| Trace region | configured size unchanged | unchanged |

Pass: every row on every config in G7's gate set (section 2). The Phase 0 spike is exempt (it needs a second
dense buffer: +364 MB at 512K, +735 MB at 1M per chip).

**G8. CI** (flip PR): the following green twice in a row with the default flipped, zero unexpected skips, expected
counts unchanged (e.g. `glm_sparse_kv_push` re-pointed at the dispatch tests: 10 correctness cases on the subtorus
carve, 3 perf, within its 30 min budget): `glm_sparse_kv_push`, the GLM sparse MLA perf job, `glm_chunked_accuracy`,
`glm53_trace`, `glm_packed`; Kimi / K3.1 jobs green with their
transport counters showing no push or dispatch calls.
Pass: all.

## 8. Steps and PRs

0. **Phase 0** (section 5) on hosts named by the user; W1, W2, W3 start here. Go / no-go to the user with numbers.
1. **Plan PR:** `compact=True`, block base ranks, owner-major multi-packet slots, per-build inbox and generation. G1
   (plan part).
2. **Op PR:** v3 only, compact landing in both writers, parts, sp_axis 1, supported set and rejections, no scratch,
   deadlock model; 3.4 if Phase 0 says so. G1, G2, G3.
3. **Integration PR:** `sparse_kv_transport` with today's behaviour as default; W1, W3, W4 land here. G4, G7.
4. **Flip PR:** default `"dispatch"` where supported; CI perf targets re-centred from this PR's CI runs (the gates
   above are the frozen ratios, independent of those targets). G5, G6, G8. Rollback drill: `"push"` on C1 and C6,
   `"gather"` on C5 pass G4 and their old perf targets.
5. **Cleanup PR:** delete `sparse_kv_push` unless (a) some mesh a model runs is supported by the push but not by the
   dispatch, or (b) any of 10 consecutive nightly runs of the G8 jobs after step 4 fails; also audit non-model callers
   of the push (tests, CI, tools) and remove them.

Every commit gets the usual two reviews (separate Claude Code CLI and Codex) before push.

## 9. Exit criteria (done = all true)

1. Phase 0 go recorded in this file, with the numbers.
2. G1–G8 pass on their applicable configs, logs linked in the PR descriptions.
3. Default transport for GLM shared layers is the dispatch on C1–C7; transport counters confirm it in CI on C1, C5,
   C6 and in the linked local logs on C2–C4, C7.
4. Rollback drill passed on C1, C5 and C6.
5. Cleanup PR merged, or step 5's (a) or (b) recorded here as the reason the push stays.

## 10. Risks

- **Planner-bound at high distinct-row counts** (L42 512K). Mitigation: 3.4; then planner work on TRISCs.
- **Lines and partial tori:** longer routes, deeper classes; Phase 0 steps 2–3 find it first.
- **Contention with sparse SDPA** on 40 cores; Phase 0 step 3 measures it in the model.
- **L1 with production tensors live** may not fit 16 planners at 1M; Phase 0 step 5.
- **sp_axis 1** is new code in every kernel that names a chip; first exercised in the op PR (G1 on C5–C7).
- **Untraced host cost:** the dispatch builds a different program per chip (core placement and routes are
  compile-time), so the host enqueues 32 programs instead of broadcasting one (+0.1–0.2 ms untraced on the gate cases).
  Productization moves per-chip data to runtime args where the kernels allow, and G5's untraced rows check it.
- **CI time:** two transports' tests until cleanup; first-run JIT per program variant. Checked in step 3.
- **Hosts:** C1, C3–C7 run on b07u08 (C5 / C6 via the stage and carve scripts); C2 on b06u08.
- **4x4 at 512K:** the whole context on 16 chips doubles per-chip KV vs 8x4; Phase 0 step 1 checks it fits.

## 11. Review log

Round 1 (v1): Codex gpt-6.2-astra (medium), Claude Code CLI; both "revise before implementation". Folded into v2:
push baseline only on C1 / C6; silent fallback to gathered tables; planner local writes need the rank lookup;
stripe-dependent entry size; owner numbering and split semantics; pre-send / parity / replay cases; memory and L1
bounds; unused scratch; contention and PREFIX / REST timing; absolute targets; fallback contradictions; staged rollout
and rollback; K3.1 decision; reproducibility and statistic; plan once, dispatch many. Not adopted: pre-send beside the
MoE (its dispatch / combine use the same fabric links).

Round 2 (v2): both "revise" (text fixes, no redesign). Folded into v3: Phase 0 as a thin integration measuring the
full -> shared -> shared sequence instead of projecting from op times (Codex B1); go thresholds equal to G5's (Claude
N2); G6 on the run that produced the capture, C1 only (Claude N1, Codex G6); forced-fabric fixtures, artifacts,
harness as work items W1–W3 (Claude N3, M8, Codex B2, 8); G8 on the flip PR only (Claude N4, Codex 10); supported set
restricted and rejection tests (Codex); per-build inbox generation (Codex 4); output spec without padding, parity
note, program-count and alternating-grid checks (Codex 5, minor); G7 comparator "today's transport", allocator
report, L1 headroom (Codex 6, Claude m1, m5); single timing decision per gate, untraced L42 limit, absolute-µs rule
(Codex, Claude §3); rollback on C1 and C5; binary cleanup condition; deadlock model ported in Phase 0
(Claude M9); Phase 0 ordered by payoff (Claude §4); one-packet guard lifted in the spike (Claude m6); watchdog env
named (Claude m3).

User, after round 2: layer performance must also be measured on the 4x4 carve. Folded in: C6 runs the real-data
layer cases (W4) with its own G5 rows, Phase 0 go row and G6 TTFT; C5 also gets the real-data cases.

User: Galaxy only for now. LoudBox (C8) and QuietBox (C9) removed from the gates; they keep today's transport (the
switch selects dispatch only on Galaxy meshes until they are qualified).
