# CCL compute helpers — first real-silicon validation (4-chip Blackhole QuietBox)

Branch: `wransom/ccl_help_compute_v2` = `wransom/ccl_hw_4chip_review` (the silicon-validated
pure-DM tip) + the compute-CCL work rebased from `wransom/ccl_help_compute` and completed here.
Every prior multi-device verification of the compute work ran on the craq-sim 8-chip Blackhole
line; this is its first run on real fabric — and the first execution anywhere of the line and
dim-zero kernel migrations, which the handoff had deliberately left unmigrated pending a
per-variant host sweep.

## Hardware

4x Blackhole **p150a** (bh-qb-13), fw bundle 19.5.0.0, ring-wired (`FABRIC_1D_RING` initializes
and passes traffic), auto-discovered mesh. All runs below on a `(1,4)` line / 4-ring with fresh
JIT compiles of the migrated kernels.

## What ran green

| Suite | Kernels exercised | Result |
|---|---|---|
| `test_reduce_scatter_async_4dev_ring -k check` (all shapes x 3 barrier variants) | migrated ring reader / writer / ring_reduction (schedule + BlockAccumulate + armed channels) | **36/36** |
| `test_reduce_scatter_async_4dev_ring -k "perf..."` subset (trace replay, 10 iters) | same, through trace capture/replay + program cache | pass |
| `test_reduce_scatter_async_line -k check` | migrated line reader / writer / line_reduction (LineSliceCursor + LineChannelWalk + SyncCadence; writer on MuxConn + split open_start/open_finish — **first execution of C.2**) | **33/33** |
| `test_reduce_scatter_async_line -k "perf..."` subset (trace replay) | same | pass |
| `test_reduce_scatter_async_training_shapes -k check` | line family, training shapes | **3/3** (9 skipped: dim-0+Linear is upstream-disabled, #26572) |
| ring dim-0 probe (`[4,1,32,256]` d0, `[8,1,32,256]` d0 → slice_B=2) | migrated dim_zero_ring reader / writer / dim_zero_ring_reduction (DimZeroChunkWalk) | **2/2** |
| all_reduce probe (`(1,4)`, 2 shapes, 2 iters) | all_reduce_async worker_writer on the **duplex Cast::Multicast stream** (arm_write + arm_fused_write_inc, write[_fused]_with_local_copy) — first silicon execution of the duplex tier | **2/2** |
| strided probe (`(1,4)`, 3 block configs, `[4,1,416,2048]` d3) | strided minimal_ring_reduction (BlockAccumulate) — **PCC 1.0 exact** (small_random_ints) | **3/3** |

Every migration also carries a committed host equivalence sweep
(`tests/ttnn/unit_tests/gtests/ccl/test_ccl_helpers_schedule.cpp`): full per-chunk traces —
indices, tile counts, wait/inc cadence flags, packet splits, every emitted tile id — diffed
against the PRE-migration kernel loops transcribed verbatim as golden. Ring: 835,584 configs /
281.9M chunk records in full mode (`TT_CCL_SCHEDULE_SWEEP_FULL=1`); line + dim-zero sweeps cover
every chip of rings 2..8, both directions, all flag combos. 0 mismatches everywhere.

## Coverage still owed (and why it cannot come from this box)

- **dim_zero_line reader/writer/reduction** — migrated + sweep-verified, but dim-0 + Linear is
  disabled in the op itself ("#26572: Can only operate on dim 3"), on every platform. First
  hardware execution has to wait for that issue.
- **deepseek_moe_reduce_scatter_reduction** — migrated + compile-clean; its kernel hardcodes
  8-slice CB arrays, so it needs an 8-device system (and its test is `@skip_for_blackhole`).
- **BlockAccumulate::rearm()** — its only caller is strided's `FUSE_RS_ADDCMUL` path, driven by
  `test_minimal_matmul_strided_reduce_scatter_async.py` (WH-gated + pinned `(1,8)`). Never
  executed anywhere yet.
- **C.1 `DuplexIncChannel` (duplex arm_inc)** and the unidirectional **`FusedWriteIncChannel`** —
  no kernel arms them yet (kept: the duplex inc is the natural "announce to both neighbours"
  primitive; the fused unicast channel is the shape a future strided/deepseek writer migration
  consumes).

## Two upstream findings (not from this effort's code)

1. `test_strided_reduce_scatter_async.py` is `@skip_for_blackhole` ("Requires wormhole_b0"), but
   strided reduce-scatter **works on Blackhole**: the (1,4) probe passes with exact PCC 1.0.
   The arch gate looks like unvalidated conservatism and is a candidate for removal.
2. The dim_zero_ring writer's pre-migration batch-ready multicast reused the barrier header but
   programmed the multicast ROUTE only under `use_barrier_sem` — with `use_barrier_sem=false`
   the batch-ready multicast went out on an unrouted header. The migration arms the barrier
   channel's route unconditionally (the same fix the dim-3 ring writer's migration made).

## One operational note

A single `--dev` (watcher) run of a `FABRIC_1D_RING` test passed but errored in fixture teardown
and left the fabric ethernet cores down; the next mesh open failed with `llrt.cpp:566: Timed out
while waiting for active ethernet core ... Try resetting the board`. `tt-smi -r` fully recovers,
and the same test without watcher passes teardown cleanly. Not reproduced since, across ~100
fabric runs.

## Round 2 — the remaining migrations (same box, same discipline)

| Migration | What moved | Verified |
|---|---|---|
| all_reduce_async compute (`reduction.cpp`) | `compute_kernel_lib::sum_blocks` — DEST-chunked via `DEST_AUTO_LIMIT` (retires `max_dst_tiles = 8`), odd block counts copy_tile-seeded (retires the empty "TODO: Future support" branch that paired blocks off the CB's end) | probe **2/2** on (1,4); a (1,3) odd-count mesh cannot init fabric on a ring-wired 4-chip box (boundary router handshakes through the excluded chip) — odd path is the silicon-verified run_seeded idiom |
| strided reader / writer / compute | writer: full fabric egress onto `FabricStreamSender<MuxConn>` + 4 armed channels (+ `MuxConn::valid()` preserving its do-nothing-without-a-link gate; + the unrouted-batch-ready fix); all three: shared neighbour-first slice cursor (`ring_neighbour_first_slice`, renamed from dim_zero_ring_first_slice); mm-block walk stays in the already-shared common header | probe extended to **5/5 configs, exact PCC 1.0** — incl. partial-last-chunk and non-divisible-Wt (ghost-tile gather + non-contiguous unicast fallback) |
| deepseek reader / writer / compute | shared slice cursor in all three; fabric egress deliberately NOT migrated — it sits on the ROUTING-PLANE stack (RoutingPlaneConnectionManager + route-id API + fused scatter-write-inc), a different fabric layer the helper does not wrap (its own helper-tier design, owed against runnable 8-dev hardware) | compile-only (op is WH-gated + 8-CB hardcoded); the cursor pattern is char-identical to the silicon-verified strided/dim-zero usages |
| llama_reduce_scatter + _create_heads compute | both reduction kernels onto `sum_blocks(pop_input=true)` — **and the handoff's open "unseeded acc_to_dest" question is RESOLVED: NOT a bug.** `tile_regs_release()`'s pack-side `llk_pack_dest_section_done` ZEROACCs the released DST region (CLR_ALL/CLR_HALF + fp32 variants, WH and BH), so DST is zero at every acquire in the standard acquire/commit/wait/release flow — the zero start these kernels rely on is a guaranteed invariant, now documented in the helper | compile-only (tests are 6U/TG); even path = the silicon-verified sum_blocks path (all_reduce probe re-passed as regression). Writer egress = the deferred C.3 shape (per-target hops + per-target direction), noted in the kernel |

## Still owed after round 2

- **rearm()** execution (strided `FUSE_RS_ADDCMUL` via the WH-gated matmul-fused test).
- First execution of: dim_zero_line kernels (#26572), deepseek + llama kernels (8-dev WH / TG systems).
- A **routing-plane helper tier** (deepseek's egress) and the **C.3 multi-target egress shape** (llama's writer) — both deliberately deferred, each needing its own design pass against runnable hardware.

### Deferred by the rebase onto `main` (`e9d0494074`)

Rebasing the 36-commit branch onto current `main` hit two upstream restructurings that
collide with this effort's migrations. In both cases the resolution was **take upstream
wholesale for the affected file** rather than hand-merge the branch's intent across the
new structure — the migration is re-done later against the new shape, as a design pass.
Exactly **three files** ended up deferred (their branch-side changes are fully dropped;
each file now matches upstream):

**(a) Ring reduce-scatter reader/writer — upstream's dual staging layout.** Upstream
restructured the ring RS staging into a dual layout that the branch's shared-schedule
migration cannot be folded into hunk-by-hunk; expressing this pair on the shared schedule
needs a **schedule-API design pass** (the schedule must be able to describe two staging
buffers, which it currently cannot).

- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduce_scatter_minimal_async_reader.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduce_scatter_minimal_async_writer.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduction.cpp` —
  added to the deferral after the post-rebase hardware retest: keeping the migrated compute
  kernel against upstream's restructured reader/writer HUNG the first ring case (upstream
  changed the trio's chunking contract together — `chunk_ring_parity` + the dual staging
  protocol). The three ring kernels form one CB contract and are deferred as a unit; the
  re-migration design pass covers all three.

**(b) V1 -> V2 fabric-mux migration obsoletes the V1 `MuxConn` egress.** Upstream migrated
this kernel to `FabricMuxV2Sender` (`tt_fabric_mux_v2_sender.hpp`), which obsoletes the
V1-`MuxConn`-based egress the branch's `FabricStreamSender` builds on. The follow-up helper
design item is a **`MuxV2Conn` connection policy** — the same connection-policy seam the
helper already has for V1, retargeted at the V2 sender — after which this writer (and any
other kernel upstream moves to V2) can be re-migrated mechanically.

- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/dim_zero_ring_reduce_scatter_minimal_async_writer.cpp`

**Not deferred, for the record:** the strided RS writer
(`minimal_ring_strided_reduce_scatter_async_writer.cpp`) was the expected next V2 casualty
but upstream has **not** moved it to V2 — its changes there are mechanical only
(`noc_obj.*`, `Semaphore`/`termination_sync.wait`, `CircularBuffer` handles, hoisted
`decompose_tile_advance`). Those compose cleanly with the branch's `MuxConn` egress and
neighbour-first cursor, so both were kept.

**Owed cleanup:** upstream's eltwise-binary init cleanup renamed `add_tiles_init` ->
`add_init` and `binary_op_init_common` -> `compute_kernel_hw_startup`, keeping the old
names as `[[deprecated]]` shims **removed on September 15th, 2026**. The CCL kernels were
migrated during the rebase, but `ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.inl`
still calls the deprecated `add_tiles_init` in 6 places (upstream never touched this
branch-new file, so nothing conflicted). It compiles today, but breaks under
`-Wdeprecated-declarations`/`-Werror` and at the removal date.

## Round 3 — the generated reference examples (same box)

The four pipeline-generated CCL ops are now committed reference examples under
`ttnn/ttnn/operations/{point_to_point,all_gather,all_reduce,reduce_scatter}` (see the
PR-review thread: examples in-tree, prompts slimmed to spec + pointers). Each package is
the exact code that graded on this box (`bh_quietbox_1x4_hw`, FABRIC_1D, fresh JIT):

| Package | Hardware grade | Notes |
|---|---|---|
| point_to_point | 383/407 golden | the 24 failures are all `non_tile_aligned` ROW_MAJOR cells, byte-identical before/after the helper consolidation — a pre-existing Phase-0 limitation of this package on Blackhole, not a helper defect |
| all_gather | 36 pass / 295 xfail-strict / 0 fail | the xfails are correct out-of-SUPPORTED refusals |
| all_reduce | 11/12 (+1 expected xfail) | |
| reduce_scatter | **29/29** golden+translated | first generated op composing all three helper families; Ring cells included |

**One helper bug found and fixed by this round** (`c7ba5604b3f`): reduce_scatter's Ring
refinement ran a fabric contract probe before implementing, and it caught that
`ccl_dm_route`'s ring wrap branch was unreachable — `fabric_1d_routing_vector` returns an
ABSOLUTE hop count, but the ring alternative was computed as if it were signed, so it was
always longer and a Ring route silently degraded to the line route (3 hops instead of 1,
wrong direction, on the (1,4) wrap pair). The fix reconstructs the signed line distance;
adjacent pairs and N/2 ties keep the line route, so no previously-green path changes.
Verified by the probe (route math, wrap connection, 1-page transfer both ways) and by the
29/29 re-grade.
