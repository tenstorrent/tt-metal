# KV-dedup un-striping: where to pay for the TP interleave

Work-in-progress notes for the dense (Kimi) KV-dedup `ag-before` path. Branched off
`ipotkonjak/kimi-kv-tp-shard-ag-before` at `e01e12abe78`.

## The problem

The TP-deduped KVPE cache is block-cyclic over `sp*tp`: chip `(s,t)` owns rows
`[c*C + s*C/sp + t*C/(sp*tp), +R)` of **every** chunk `c`, with `R = C/(sp*tp)`.
`_gather_kvpe_tp_to_sp` rebuilds one SP rank's slab before `ring_mla`.

Two ways to get that slab into the chunk-major order `ring_mla` reads:

1. **`gather`** (what the branch does today) — pass `input_stripe_size=R`, so the gather
   interleaves the stripes and hands `ring_mla` the layout it always read.
2. **`reader`** — let the gather concatenate the shards rank-major (its fast path) and have
   `ring_mla`'s reader decode back to natural order.

Selected at runtime by `TT_MLA_KV_DEDUP_UNSTRIPE=gather|reader` (`mla.py`), plus a `none`
diagnostic (plain gather, no decode) that must be wrong by construction.

## Why `reader` is worth having: the gather's stripe cost is large

Measured with `tests/ttnn/unit_tests/operations/experimental/test_tmp_stripe_util.py`
(8x4, `cluster_axis=1` (TP), 2 links, 1152 B pages, FABRIC_2D). Both arms move the same bytes —
with one stripe `gathered_dim_size` narrows the front of the shard, with several it bounds whole
stripes, and both land on `gathered_dim_size/tp` active rows per device.

| per-link bytes | contiguous (1 stripe) | striped (8 stripes) | delta |
| --- | --- | --- | --- |
| 5.31 MB (above the 1.5 MB bank-owned floor) | 0.3772 ms / 28.14 GB/s | 0.5413 ms / 19.61 GB/s | **-30.3%** |
| 0.885 MB (below it) | 0.2025 ms / 8.74 GB/s | 0.2164 ms / 8.18 GB/s | -6.4% |

The two operating points decompose the loss: below the byte floor both arms run 2
workers/direction, so the ~6% is the **bank-owned schedule** alone; above it the contiguous arm
also picks up the **4-worker tier** and the gap widens to ~30%.

`single_stripe` hard-gates `can_use_bank_owned`, and on Blackhole the 4- and 8-worker tiers are
both conditioned on bank ownership. The only non-bank-owned escape to more workers is
`use_high_parallelism_fallback`, which requires `input_page_size >= 2048`; the KVPE page is
1152 B, so **a striped gather is pinned to 2 workers/direction at any message size**.

## Option B (stripe-aware bank-owned gather) is dead for the real geometry

Idea: relax the `output_chunks_per_stripe == num_input_pages` eligibility and teach the
bank-owned writer the stripe jump, keeping chunk-major output *and* the fast schedule — no
`ring_mla` change at all.

It fails its precondition. The bank-owned schedule maps input page -> output page as the
**identity** (`local_output_start == input_page_start`). With stripes it becomes
`output(p) = p + (p/R)*R*(num_devices-1)`, so bank residency survives only if
`pages_per_stripe * (num_devices-1) ≡ 0 (mod num_dram_banks)`. For the real cache (TILE layout,
`kv_lora_rank 512 + qk_rope_head_dim 64 = 576` -> 18 width tiles, `rows_dev = 640/4 = 160` rows
-> 5 seq tiles):

```
pages_per_stripe = 5 * 18 = 90
90 * (4-1) = 270,  270 mod 8 = 6  != 0        (Blackhole has 8 DRAM banks)
```

A worker's output bank would shift by 6 at every stripe crossing. The stripe is fixed by the
cache layout, so it cannot be padded to a multiple of 8. **Do not re-derive this from a
ROW_MAJOR harness** — with page == row and a 512-page stripe the condition *does* hold, which is
how this was briefly and wrongly thought viable.

## Option A: implemented, decode provably correct, still fails

`BlockCyclicPaddedAddrGenerator` (`dataflow_common.hpp`) splits a read at stripe boundaries and
issues one contiguous strided block per segment, base tile id via the existing
`tt::block_cyclic` invP. Key property: the permutation is **piecewise contiguous at stripe
granularity** (a run of `R` logical tiles is also a run of `R` physical tiles), so the per-tile
NoC read count is unchanged — `issue_block_reads` already issues one `async_read` per tile.
`BlockCyclic == false` folds back to a single unremapped read, so every other caller is
bit-identical.

Plumbing: `kv_block_cyclic_stripes`/`_ranks` on `RingJointSDPAParams` (hashed, defaulted 1/1)
-> factory derives `stripe_rows_t = kv_local_padded_Nt/(stripes*ranks)` and emits reader compile
slots 44-47 (**tensor accessors moved 44 -> 48**) -> device-op invoke -> `ring_mla` public entry
-> nanobind.

### Status: WRONG, and not for any reason found so far

`test_mla_chunked_prefill_tp_shard_kv`, `tp_sharded`, `torus-xy-8x4`:

| mode | result |
| --- | --- |
| `gather` (both arms, 8 cases) | **passes** |
| `none` (plain gather, no decode) | out PCC 0.5042 |
| `reader` (decode local + gathered) | out PCC **0.8196** |
| `reader`, decode local slab only | out PCC 0.6375 |
| `reader`, `Sk_chunk_t=5` (one stripe/read, no segmentation) | out PCC 0.8196 |

Fails at `test_mla.py:906` (per-iteration **output** PCC, threshold 0.98) on **iteration 0**.
`deep-20k` is the one passing case; it preloads 20k so `active_chunks=5`, versus `1` for the
failures. The KV cache PCC itself stays healthy (0.9998) — only attention output is wrong.

### Ruled out (all verified, not assumed)

- **Gather layout.** `test_tmp_layout_probe.py` confirms rank-major with a full-shard per-rank
  stride, 16/16 across `{ROW_MAJOR, TILE} x active_chunks{1,3} x links{1,2} x slots{1,3}`.
  The striped gather equals natural order bit-exactly.
- **Reader addresses.** DPRINT from the generator, all 8 SP slabs:
  `d2=r*80 -> phys = r*80 + t*20` for `row=0,5,10,15`, `run=5`, `vr=20`, with `R=5 S=4 N=4`.
  Exactly the derivation `t*20 + c*5`; compile-time args arrive correctly; only chunk 0 is read,
  as the mask requires.
- **invP arithmetic**, and its equivalence to the indexer's proven formulation
  (`bc_sp = sp*split`): `(s*tp + t)*(n_chunks*R) + c*R == s*80 + t*20 + c*5`, identical to the
  per-slab decomposition.
- **Segmentation** — forcing one stripe per read gives the identical PCC.
- **Gathered buffer needs the decode** — removing it made things worse (0.6375).
- **Attend logic.** `kv_pad_rotation_enabled` is on (mla.py passes `kv_actual_isl`), so the
  active mask is the per-column path at `compute_streaming.hpp:1035`, whose contract is
  "column `col` is local tile `k_local_start_tile + col`" — satisfied, since the CB is filled in
  logical order. Verified numerically for iteration 0: valid logical 0-19 maps to physical
  {0-4, 20-24, 40-44, 60-64}, exactly the region the gather wrote.
- **V** — `kt_inplace_v` needs `Sq_chunk_t == 1` and MLA chunked uses `q_chunk_size=32`, so V is
  read from K^T in L1, never from DRAM. The V-generator remap is inert.
- **Reader/AG race** — `get_next_ring_id_and_sync` waits per *ring id*, i.e. until slab `r` has
  fully landed; the decode never leaves a slab.
- **The writer** — builds only output/joint-output addresses, never reads the KV slab.
- **The fused AG copy** — verbatim linear (`tile_id = base + tiles_read - input_origin_page`).

Every component is individually correct and the composition is still wrong, so the hole is in
the *model of the composition*, not in any one piece.

### Next step

Differential data rather than more hypothesis-testing: DPRINT the **K tile values** at a known
address in both `gather` and `reader` mode.

- same addresses, different data -> localizes to the buffer contents
- same data, wrong result -> localizes downstream of the read

Unlike every probe above, that cannot come back "consistent".

## Temporary scaffolding in this commit (remove before review)

- `KV_DEDUP_UNSTRIPE_MODE` `none` diagnostic and `TT_MLA_FORCE_K_CHUNK` override (`mla.py`)
- `BCPROBE` `log_warning` in `ring_joint_sdpa_program_factory.cpp`
- `BCP` DPRINT block + `api/debug/dprint.h` include in `dataflow_common.hpp`
  (note: the stream API `DPRINT << ... << ENDL()` is **removed**; use fmt-style
  `DPRINT("x={}\n", x)`, and deprecation is an error under `-Werror`)
- `tests/.../test_tmp_stripe_util.py`, `tests/.../test_tmp_layout_probe.py`

## Environment notes

- Build **only** with `./build_metal.sh --build-dir build_Release`. A narrow
  `cmake --build --target ttnn` left `_ttnncpp.so` stale against an edited program factory; the
  JIT kernel then had the new compile-arg layout while the host emitted the old one, producing
  misleading accessor `static_assert`s and `'chains_base_offset' is not captured`. Host and JIT
  kernel must rebuild together.
- Box: 32-device Blackhole Galaxy (8x4). Torus wrap links are unstable — prefer `FABRIC_2D`;
  torus runs did hold for these tests but are not dependable.
- Long pytest runs must be detached (`setsid nohup`): the harness kills tracked background tasks
  when `free` memory dips during weight load, even though `available` stays ~500 GB.
