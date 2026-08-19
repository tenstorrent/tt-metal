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
