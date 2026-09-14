# KV-dedup un-striping: where to pay for the TP interleave

Notes for the dense (Kimi) KV-dedup `ag-before` path. Branched off
`ipotkonjak/kimi-kv-tp-shard-ag-before` at `e01e12abe78`.

**Status: the `reader` arm is correct and is the faster of the two.** The root cause of the long-standing
PCC failure is found, fixed and measured; what remains is cleanup and wider regression. See
[Root cause](#root-cause-the-fused-ags-prefix-bound-slices-the-wrong-axis).

## The problem

The TP-deduped KVPE cache is block-cyclic over `sp*tp`: chip `(s,t)` owns rows
`[c*C + s*C/sp + t*C/(sp*tp), +R)` of **every** chunk `c`, with `R = C/(sp*tp)`.
`_gather_kvpe_tp_to_sp` rebuilds one SP rank's slab before `ring_mla`.

Two ways to get that slab into the chunk-major order `ring_mla` reads:

1. **`gather`** — pass `input_stripe_size=R`, so the gather interleaves the stripes and hands
   `ring_mla` the layout it always read.
2. **`reader`** — let the gather concatenate the shards rank-major (its fast path) and have
   `ring_mla`'s reader decode back to natural order.

Selected at runtime by `TT_MLA_KV_DEDUP_UNSTRIPE=gather|reader` (`mla.py`).

## Why `reader` is worth having: the gather's stripe cost is large

Measured on the standalone TP gather (8x4, `cluster_axis=1` (TP), 2 links, 1152 B pages, FABRIC_2D).
Both arms move the same bytes.

| per-link bytes | contiguous (1 stripe) | striped (8 stripes) | delta |
| --- | --- | --- | --- |
| 5.31 MB (above the 1.5 MB bank-owned floor) | 0.3772 ms / 28.14 GB/s | 0.5413 ms / 19.61 GB/s | **-30.3%** |
| 0.885 MB (below it) | 0.2025 ms / 8.74 GB/s | 0.2164 ms / 8.18 GB/s | -6.4% |

Below the byte floor both arms run 2 workers/direction, so the ~6% is the **bank-owned schedule** alone;
above it the contiguous arm also picks up the **4-worker tier** and the gap widens to ~30%.

`single_stripe` hard-gates `can_use_bank_owned`, and on Blackhole the 4- and 8-worker tiers are both
conditioned on bank ownership. The only non-bank-owned escape is `use_high_parallelism_fallback`, which
requires `input_page_size >= 2048`; the KVPE page is 1152 B, so **a striped gather is pinned to 2
workers/direction at any message size**.

Note this is the *standalone* TP gather, which is fully exposed. It is a different transfer from the
*fused* AG inside `ring_mla`, which overlaps attention compute — a distinction that matters below.

## Option B (stripe-aware bank-owned gather) is dead for the real geometry

Idea: relax the `output_chunks_per_stripe == num_input_pages` eligibility and teach the bank-owned writer
the stripe jump, keeping chunk-major output *and* the fast schedule.

It fails its precondition. The bank-owned schedule maps input page -> output page as the **identity**
(`local_output_start == input_page_start`). With stripes it becomes `output(p) = p + (p/R)*R*(num_devices-1)`,
so bank residency survives only if `pages_per_stripe * (num_devices-1) = 0 (mod num_dram_banks)`. For the
real cache (TILE, `kv_lora_rank 512 + qk_rope_head_dim 64 = 576` -> 18 width tiles, `rows_dev = 640/4 = 160`
rows -> 5 seq tiles):

```
pages_per_stripe = 5 * 18 = 90
90 * (4-1) = 270,  270 mod 8 = 6  != 0        (Blackhole has 8 DRAM banks)
```

**Do not re-derive this from a ROW_MAJOR harness** — with page == row and a 512-page stripe the condition
*does* hold, which is how this was briefly and wrongly thought viable.

## Option A: `BlockCyclicPaddedAddrGenerator` — correct all along

`dataflow_common.hpp` splits a read at stripe boundaries and issues one contiguous strided block per
segment, base tile id via the existing `tt::block_cyclic` invP. The permutation is **piecewise contiguous
at stripe granularity**, so the per-tile NoC read count is unchanged. `BlockCyclic == false` folds back to
a single unremapped read.

This decode was never wrong. Proof: once the AG bug below was fixed, the `reader` arm's output became
**bit-identical to the `gather` arm on every iteration** (see [Verification](#verification)).

## Root cause: the fused AG's prefix bound slices the wrong axis

`compute_gather_valid_Ht` (`ring_joint_sdpa_program_factory.cpp`, and the on-device twin in
`ring_attention_all_gather_metadata.hpp`) bounds the fused all-gather to a **contiguous per-device page
prefix**:

```
valid_slabs = ceil(logical_n / chunk_global)
return valid_slabs * chunk_local_tiles
```

That is correct only when the populated chunks sit at the front of each device's slab — true in
chunk-major, **false in rank-major**. In rank-major, tiles `[0, A*20)` are tp ranks `0..A-1` *in their
entirety* (all their chunks, populated or not), and tp ranks `A..3` ship nothing. The reader then computes
correct addresses into tiles the ring never wrote.

How it was found: slicing the output into 160-token partitions (one `(chunk, sp, tp)` stripe) at
`active_chunks=1` gave a clean step — **sp 0 correct across all four tp slots (~0.998), sp 1-7 broken
(0.55-0.78)**. sp 0 is read from local DRAM and never crosses the ring; everything that crosses the ring
was short. `test_mla.py`'s `_log_stripe_partition_pcc` (`TT_MLA_STRIPE_PCC=1`) prints that table.

Corroborating: `fullchunk-1u` runs `active_chunks` 1,2,3,4 across its four iterations, and PCC was
0.8196 / 0.7998 / 0.9183 / **0.9899 pass** — iteration 3 is where the prefix finally spans the slab.
`deep-20k` passed for the same reason (`logical_n ~ 25600` saturates `valid_slabs`).

Why every earlier probe came back clean: the addresses *were* right. The prior evidence table listed
"the fused AG copy" as ruled out, but it checked the copy's **addressing**, not its **extent**.

### Dead hypothesis, recorded so it is not re-derived

The prefix bound does **not** shrink the per-rank output stride.
`high_bw_all_gather_device_operation_types.hpp:57`: "each rank's local prefix occupies that rank's
**fixed worst-case output slot**, and the allocation remains full size." Rank `t` always lands at
`t * full_shard`; `gathered_dim_size` only limits how much of that slot is written.

## The fix: move the right rows, not more rows

`BlockCyclicRowMap` (`ring_attention_all_gather_metadata.hpp`) — the populated rows of a rank-major slab
are `ranks` runs at pitch `stride`, not one prefix:

```
seg    = gather_valid_Ht / ranks          // rows transferred per rank
stride = cache_local_tile_rows / ranks    // rank block pitch
physical_row(L) = (L / seg) * stride + (L % seg)
```

The transferred **count is unchanged** (`ranks * seg == gather_valid_Ht`), so the link split and the
producer/consumer page-count protocol are untouched; only the row positions move. `ranks <= 1` is the
identity, so every existing caller is bit-identical. Both kernels walk the *logical* row space and place
each row via `physical_row`; they must agree or destination offsets diverge.

Derived from `input_valid_pages / Wt` and `input_tensor_Ht` so it serves the scalar and metadata paths
alike.

Plumbed as `kKvBlockCyclicRanks`, appended to both kernels' compile args (fixed counts 23->24 and 24->25),
through the AG factory signature (defaulted to 1) from `RingJointSDPAParams::kv_block_cyclic_ranks`.

### The bank-owned schedule is disabled when `ranks > 1`

`prefetch_bank_owned_slices` maps input page -> output page as the identity to keep each worker on its own
bank; a multi-run transfer cannot preserve that. Expressing it would need one bank-owned call per rank
block, which changes per-link page accounting on both ends. This mirrors `high_bw_all_gather` gating
`can_use_bank_owned` on `single_stripe`.

The gate is compile-time, so it stays off even at full population where `seg == stride` makes the map the
identity. The measurements below show the reader arm winning **while carrying that handicap**, which is
the evidence that the fused AG's schedule loss hides behind attention compute. Narrowing the gate (it
would have to be correct for every extent the program might see, and `gather_valid_Ht` varies per
dispatch on the metadata path) is the remaining perf idea, and the numbers say there is little left in it.

## Verification

`test_mla_chunked_prefill_tp_shard_kv`, `tp_sharded`, `torus-xy-8x4`, `fullchunk-1u`:

| iter | active_chunks | reader (fixed) | gather |
| --- | --- | --- | --- |
| 0 | 1 | 0.9972156523970509 | 0.9972156523970509 |
| 1 | 2 | 0.9930977231449342 | 0.9930977231449342 |
| 2 | 3 | 0.9912190184107640 | 0.9912190184107640 |
| 3 | 4 | 0.9899478731558230 | 0.9899478731558230 |

Bit-identical on every iteration — what a pure data-movement fix should produce.

Full suite in `reader` mode: **8 passed** on `torus-xy-8x4` (all four `sp_only` cases, i.e. the
`ranks == 1` identity check, and all four `tp_sharded`). The 8 `fabric2d-2x4` cases errored, including
`sp_only` ones where this change is inert and whose `torus-xy-8x4` twins pass — believed to be the 2x4
submesh on this 8x4 Galaxy, **not yet verified**.

## Perf: op-level, kimi_k2_7 50k+5k, 8x4

`test_mla_chunked_tp_shard_kv_perf`, whole-forward time:

| arm | samples | mean | vs sp_only |
| --- | --- | --- | --- |
| sp_only (standard ring_mla, no dedup) | 7.233, 7.212 | **7.223 ms** | — |
| tp_sharded + gather | 7.523, 7.384 | **7.454 ms** | +231 us, **+3.2%** |
| tp_sharded + reader | 7.306, 7.295, 7.297 | **7.299 ms** | +76 us, **+1.05%** |

**TP dedup costs ~1% of forward time on the reader path vs ~3% on the striped gather, for 4x less KV
cache DRAM.** The reader arm is also the more stable (11 us spread over 3 runs, against the gather arm's
139 us over 2) — consistent with the striped gather being pinned to 2 workers/direction on the general
schedule. Treat the gather arm's +3.2% as approximate (n=2, range +2.4% to +4.0%).

At this depth the cache is fully populated, so the remap is the identity and the fix contributes nothing
to these numbers; they measure the reader path's gather win alone.

## Remaining work

- Strip the scaffolding: `TT_MLA_STRIPE_PCC` / `_log_stripe_partition_pcc` and `TT_MLA_PCC_SOFT`
  (`test_mla.py`), the `none` diagnostic mode and `TT_MLA_FORCE_K_CHUNK` (`mla.py`), the `BCPROBE`
  `log_warning` (`ring_joint_sdpa_program_factory.cpp`), the `BCP` DPRINT block (currently `if (false)`)
  and its `api/debug/dprint.h` include (`dataflow_common.hpp`), and `tests/.../test_tmp_stripe_util.py` /
  `test_tmp_layout_probe.py`.
- Run the shared-AG regressions: `tests/nightly/tg/ccl/test_ring_attention_all_gather.py` and
  `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` (the `ranks == 1` identity paths).
- Verify the `fabric2d-2x4` errors are environmental.
- Decide whether `reader` becomes the default and `TT_MLA_KV_DEDUP_UNSTRIPE` goes away.

## Environment notes

- Build **only** with `./build_metal.sh --build-dir build_Release`. A narrow `cmake --build --target ttnn`
  left `_ttnncpp.so` stale against an edited program factory; the JIT kernel then had the new compile-arg
  layout while the host emitted the old one, producing misleading accessor `static_assert`s.
- Box: 32-device Blackhole Galaxy (8x4), high-power.
- **Never SIGKILL a device test.** `kill -9` skips pytest's teardown (`Closing user mode device drivers`
  -> `Closing devices in cluster` -> cluster destructor), leaving the cluster half-initialised. The next
  run then blocks in `do_poll` at 0% CPU partway through model build -- a hang, not a crash, so it burns
  the full timeout. SIGTERM and let teardown finish; escalate only if it will not exit, and let the
  devices settle before relaunching. This is also the mechanism behind the torus skip below, and the
  likely cause of the SIGBUS-during-weight-load crashes seen after earlier kills.
- The torus skip in `models/demos/deepseek_v3_d_p/tests/conftest.py:277` is **intermittent for this
  reason**: `skip_rings` defaults to true when `ttnn.cluster.get_cluster_type()` throws, and that call
  opens the cluster. Every skip observed landed immediately after a killed or crashed run; retrying on
  settled devices worked. A single skipped perf run is not evidence of a gate.
- Distinguish hang from work with `top -H`, NOT `ps -o %cpu`. `ps` reports CPU time / elapsed since
  process start, so a run that compiled hard and then wedged still shows thousands of percent -- it read
  as "busy compiling" for 20 minutes while every one of its 296 threads sat at 0.0%. A real JIT
  repopulate also spawns compiler children (`sfpi-*`, `cc1plus`) and writes artifacts; check for those.
- Long pytest runs must be detached (`setsid nohup`), and **not** under the session scratchpad in `/tmp` —
  it was swept mid-run and took the logs with it. `generated/` is gitignored and durable.
