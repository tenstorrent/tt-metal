# KV-dedup un-striping: where to pay for the TP interleave

Notes for the dense (Kimi) KV-dedup `ag-before` path. Branched off
`ipotkonjak/kimi-kv-tp-shard-ag-before` at `e01e12abe78`.

**Status: the `reader` arm is correct, `tp_shard_kv` is trace-safe, and TP dedup is FREE under trace**
-- 4x less KV cache DRAM for -0.26% of model forward time (noise). The root cause of the long-standing
PCC failure is found and fixed; the traced path, previously refused outright, now works. What remains is
cleanup and wider regression. See [Root cause](#root-cause-the-fused-ags-prefix-bound-slices-the-wrong-axis)
and [Trace safety](#trace-safety-tp_shard_kv-under-a-captured-program).

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

## Trace safety: `tp_shard_kv` under a captured program

Previously refused outright (`assert metadata is None` in `_chunked_attn`). Two pieces of per-chunk
state had to reach the device instead of being frozen into the capture.

**Slot.** `high_bw_all_gather` already had `input_batch_index_tensor` plus `batch_slot_num_layers` /
`batch_slot_layer_idx`, so the gather recomposes `user*layers + layer_idx` on-device exactly as
`ring_mla`'s readers do. The old refusal blamed the `[B, n_chunks, R, W]` view, which commit
`0b93b2c1cd8` removed -- the gather reads the cache in place now, so that reason was stale.

**Extent.** `gathered_prefix_tensor` wants the chunk start in the GATHERED dim's units. For the SP-axis
gathers (the sparse path, `mla.py:2156`) that is global tokens and `metadata[1]` works directly. This
gather rides the TP axis, so its dim is one SP rank's slab -- `global/sp` -- and the same scalar is
`sp` times too large. New hashed `gathered_prefix_divisor` (default 1) divides it at the single point
every extent derivation flows from (`unicast_reader.cpp`, `gathered_dim_size_for_prefix`). Integer
division is exact enough: it shifts the start by less than one element and the slab is at least one, so
the round-up to whole slabs is unaffected even on rotated partial chunks.

The alternative -- what the sparse `sp == 1` fallback does (`mla.py:2380`) -- is pinning both extents to
the full buffer: chunk-invariant and therefore safe to bake, but it moves the whole cache every chunk on
all 61 layers. Fine for a QuietBox fallback, not for the main 8x4 path.

### `has_metadata()` split, and the two bugs it exposed

`ring_mla` then rejected the batch-1 slab: *"K cache batch=1 must be divisible by
kv_cache_num_layers=61"*. `has_metadata()` required `slot_id` AND `kv_actual_isl`, conflating two
independent things -- `kv_actual_isl` is the trace-safe core every captured chunk needs, while the slot
only matters to a caller that has not already chosen one. A KV-deduped caller spends its slot in the TP
gather and hands over a batch-1 slab: extent yes, slot no.

Split into `has_metadata()` (extent) and `has_slot_metadata()` (slot). The layer-divisibility check now
gates on the slot -- it exists only to recompose `slot*layers + idx` -- and validation's
`has_indexed_kv_cache` uses the same rule as the factory's `slot_from_metadata`, so both describe the
same program.

That split exposed two places keyed on the wrong flag. **Both HUNG the reader rather than failing it**,
which is why they cost so much to find:

1. the `kv_actual_isl` **accessor** was appended inside the slot block, so the kernel was compiled
   expecting an accessor the host never pushed and read misaligned compile args;
2. the reader's **common runtime args** were pushed only under the slot -- but `kv_actual_isl`'s address
   is index 4 of that block, read whenever `kv_pad_from_metadata`, so the reader dereferenced garbage.

The writer had it right already, gating on `kv_pad_from_metadata` alone; the reader's nesting was the
anomaly.

### Test coverage this needed

`use_metadata_tensor` is now an axis on `test_mla_chunked_prefill_tp_shard_kv`. It is the ONLY cover for
the TP gather's on-device slot select and its divisor extent, and both fail **silently** in a perf job
(`check_pcc=False`): a baked slot reads another user's KV, a short gather leaves the tail unpopulated.
`fullchunk-2u` varies the slot (u0 0.99721565, u1 0.99721338), `deep-20k` grows the prefix (0.9985742 at
`kv_actual=20480`). 4/4 on both `tp_sharded` and `sp_only`.

Note the structural gap that let the batch-1 bug reach a 25-minute L61 run: **the MLA unit test
allocates one cache layer**, so `1 % 1 == 0` and the layer factor is unreachable there. An L10-scale
case is the cheapest place it is real.

## Perf: model-level, kimi_k2_7 L61, TRACED (the headline)

`test_kimi_prefill_transformer_chunked_perf`, 11 chunks x 5120, 10 iters, torus-xy-8x4, reader mode.
A `tp_shard_kv` axis was added to that test -- no Kimi test had one ("tp_sharded has no CI job on either
side").

| chunk | sp_only | tp_sharded | delta |
| --- | --- | --- | --- |
| 0 | 0.422 | 0.421 | -0.001 |
| 1 | 0.430 | 0.432 | +0.002 |
| 2 | 0.463 | 0.462 | -0.001 |
| 3 | 0.489 | 0.489 | 0.000 |
| 4 | 0.521 | 0.519 | -0.002 |
| 5 | 0.553 | 0.551 | -0.002 |
| 6 | 0.581 | 0.579 | -0.002 |
| 7 | 0.611 | 0.609 | -0.002 |
| 8 | 0.657 | 0.654 | -0.003 |
| 9 | 0.693 | 0.691 | -0.002 |
| 10 | 0.731 | 0.728 | -0.003 |
| **total** | **6.151 s** | **6.135 s** | **-16 ms, -0.26%** |

**TP dedup is free under trace.** Per-chunk stddev is <= 0.002 s and every delta is <= 0.003 s in both
directions, so this is "indistinguishable from baseline", not "slightly faster". n=1 per arm.

The one gate FAIL (chunk 1, 0.432 against a band top of 0.431) is not a regression: `sp_only` measured
0.430 on that same chunk, so both sit on the band edge, and the recorded baselines are `sp_only`-derived
and do not apply to this arm anyway.

### Do not quote the notrace number

The notrace arm measured **+6.1%** (0.9228 vs 0.8695 s/chunk) and that figure is an artifact. Notrace is
host-dispatch-bound: chunk time is FLAT at ~0.87 s regardless of KV depth, and ~36% of it is host
dispatch. A flat per-layer delta there mostly measures dispatching one extra gather op per layer. Traced
is device-bound -- chunk time rises 0.422 -> 0.731 s as the cache fills -- which is the regime that
answers the question. The same box fails the notrace gate by ~7% while PASSING the tighter traced gate,
for the same reason: its host is slow, its devices are on-baseline.

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
  `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` (the `ranks == 1` / `divisor == 1` identity
  paths, plus the GLM sparse path that shares `high_bw_all_gather`).
- Record a `tp_sharded` perf baseline so that arm gates instead of reporting; today it is compared
  against `sp_only`-derived numbers.
- Decide whether the `tp_shard_kv` axis added to `test_kimi_prefill_transformer_chunked_perf` earns a
  CI job, and revert the `fabric_2d_line` param added to `test_high_bw_all_gather_galaxy_ci_perf` for a
  one-off torus-vs-line comparison (torus 85.6 GB/s vs line 47.2 GB/s on the shared shape, 1.82x).
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
