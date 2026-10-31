# Decode SDPA: improvements

Status: plan. Measured on `dsv41_decode_32k_32L` (32 layers, decode at position ~32k, 8 stages x TP4
on Blackhole, batch 1): `generated/profiler/reports/dsv41_decode_32k_32L/2026_10_30_08_39_55/`.
All times are device kernel durations from `ops_perf_results_*.csv.gz` (the run is eager, so op-to-op
gaps are left out); "per token" sums the 32 layers. At 36 tok/s/u a token is 27.8 ms.

Companion: `DECODE_LIGHTNING_INDEXER_IMPROVEMENTS.md` (the select that fills the KV SDPA reads).

## 1. Where the time goes

Of the 32 layers, 0-1 are sliding-window only and 2-31 compressed. Every compressed layer attends
`[its 128-row ring | the group's 512 selected rows]` = 640 keys through
`flash_multi_latent_attention_decode` (`_sdpa` in `tt/decode/attention.py`), K == V, 16 local heads,
`head_dim` 512.

| op | per call | calls / token | per token |
|---|---|---|---|
| `SdpaDecode`, compressed layer (640 keys) | 68.8 us | 30 | **2.06 ms** |
| `SdpaDecode`, sliding layer (128 keys) | 21.5 us | 2 | 0.04 ms |
| output RoPE (`FusedPartialRope`, **1 core**) | 23.0 us | 32 | **0.74 ms** |
| ring copy into `sel_tile` (`fill_cache` -> `UpdateKVCacheOperation`, 4 cores) | 7.1 us | 30 | 0.21 ms |
| `sel_kv` -> `sel_tile` (`Tilize`, 20 cores), index layers only | 7.6 us | 6 | 0.05 ms |
| q replicas around SDPA (`AllGatherForMatmul` x2, q to DRAM and back in index layers) | ~2.5-5 us | 32 | ~0.1 ms |
| **total** | | | **~3.2 ms** (11% of the token) |

## 2. Why one SDPA takes 69 us

The op's parallelism is `B x num_kv_heads x cores_per_head` (`sdpa_decode_program_factory.cpp`,
"Core Allocation"). Decode here is one user with one shared K == V head, so the only split is
along the keys, and `max_cores_per_head_batch = 2` (`configs/system_configs.yaml`,
`AttentionSettings`) caps it at **two cores**. The profile's "120 cores" is the program grid, not the
cores doing work. Each of the two walks 320 keys in 32-key chunks: per chunk 16 tile matmuls for
`Q K^T` (inner dim 512) and 16 for `P V` (512 output columns), at HiFi4.

The two layer types fit one line well:

    t(us) ~ 9.7 + 0.185 * keys_per_core        (64 keys/core -> 21.5, 320 keys/core -> 68.8)

0.185 us per key moves 1 KB of K per key: ~5.5 GB/s per core, far below what a core's NoC reads.
The op is **compute-bound on two cores**. Q already uses half tiles (16 heads <= 16,
`use_half_tile`), so the per-chunk work does not shrink with the head count. The lever is the number
of cores per head, then the cost of each chunk.

## 3. Changes

### 3.1 More cores per head, by fixing the tree-reduction scratch (largest win)

The YAML records why it is 2: *"the per-core reduction scratch CB grows linearly in
(cores_per_head - 1), and at 4 it runs past the resident projection weight and fails to build."*

That linear growth is a sizing bug, not a property of the reduction. The factory sizes `c_19`
(the children's partial outputs) as

```cpp
// sdpa_decode_program_factory.cpp:443
const uint32_t intermed_output_tiles = (out_tiles + 2 * PNHt) * (num_cores_per_head - 1);
```

but the reduction is a tree: a child writes its block at `send_at_round * (out_chunk_tiles + 2 * PNHt)`
of its parent's `c_19` (`writer_decode_all.cpp:331`), and the parent reads round `r`'s child at
`r * (...)` (`writer_decode_all.cpp:267`). Only `num_tree_reduction_rounds = ceil(log2(n))` slots
are ever addressed. Sizing it `(out_tiles + 2 * PNHt) * num_tree_reduction_rounds` is
enough. Each slot here is 16 output tiles plus 2 statistics tiles.

| cores / head | slots today | slots after the fix | keys / core | projected SDPA |
|---|---|---|---|---|
| 2 (today) | 1 | 1 | 320 | 69 us |
| 4 | 3 (fails to build) | 2 | 160 | ~40 us |
| 8 | 7 | 3 | 80 | ~25-28 us |
| 16 | 15 | 4 | 40 | ~18-22 us |

Projections take the fit of section 2 and add ~1-2 us per extra tree round, which still needs measuring.
After the fix, 8 cores costs the L1 that 4 costs today, so check whether 3 slots fit next to
`keep_weights_in_l1`. Two more L1 reliefs, both worth checking in the factory:

* With `use_mla` the reader builds each V chunk from the K chunk already in L1 (the `_sdpa` comment:
  "KV is read once"). If `c_2` (`v_tiles = Sk_chunk_t * vDHt * 2`, 64 KB at chunk 32) is still
  allocated in that mode, it is dead space.
* Only active cores need the big CBs, but they are created on the whole grid. Allocations are
  lock-step across cores, so this frees nothing on its own. It matters only together with
  `sub_core_grids`, which would put SDPA on cores whose L1 is not under the resident weights.

Steps: change line 443, run `tests/ttnn/unit_tests/operations/sdpa/test_mla_decode.py` at 1/2/4/8/16
cores per head, then the 32L profile with `DEEPSEEK_V4_SDPA_MAX_CORES_PER_HEAD=4`, `8` and `16`.
**Expected: -1.2 ms/token at 8 cores.**

### 3.2 Cheaper chunks: fidelity and chunk size

* `_HIFI4_SDPA` (`tt/common.py`) is HiFi4 for both matmuls. bf16 x bf16 at HiFi2 is the usual
  attention setting. Make the fidelity an `AttentionSettings` field (`sdpa_math_fidelity`, env
  `DEEPSEEK_V4_SDPA_FIDELITY`) and compare HiFi2 and LoFi against HiFi4 in
  `tests/decode/test_attention.py` and by end-to-end text. Keep `fp32_dest_acc_en=False`: the
  comment above `_HIFI4_SDPA` records it producing garbage for this shape.
* `sdpa_exp_approx_mode` is `false` in the profile (the op's own default is `true`). Measure it
  with the fidelity change.
* `k_chunk_size` 32 -> 64 halves the per-chunk fixed cost (one `Q K^T` and softmax-update round per
  chunk) and doubles the K buffer (`k_tiles = Sk_chunk_t * DHt * 2`: 64 -> 128 KB). It competes with
  3.1 for L1, so tune it after 3.1.

The 9.7 us intercept is fixed per call, and with 3.1 it becomes most of the op. Precision then buys
roughly a third of the remaining slope: about -0.1 to -0.3 ms per token, to be measured.

### 3.3 Spread or fuse the output RoPE (23 us on one core)

`_attend` hands the SDPA output, `[16, 512]` height-sharded on one core (the single reducer:
`_sdpa_decode_output_config`), straight to `_apply_rope(attn, cos, neg_sin, ...)`. The op then
walks all 256 `1x32` tiles of `[1, 8192]` on that core, to rotate 64 of every 512 columns. The q
RoPE does the same work on 64 cores in 4.8 us.

* **A (small):** reshard the output to 16 cores, one head each (or width-shard it as the q RoPE's
  input is), before `_apply_rope`. The reshard costs ~1-2 us. Expected: 23 -> ~5 us.
* **B (better):** apply it in the SDPA reducer. The reducer already holds the normalized output
  tiles before it untilizes them. The rotation touches only the last two 32-column tiles of each
  head and is the same per head (one position), so it is two small tile ops against a `cos`/`sin`
  row the reader fetches once. That removes an op and an L1 round-trip.

`docs/fuse_partial_rope_into_matmul_decode.md` (V4) leaves this RoPE out as "not fusible". It is the
largest of the three: q is 4.8 us on 64 cores, kv 2.5 us, the output 23 us. **Expected:
-0.6 ms/token with A, a little more with B.**

### 3.4 Stop copying the ring into the shared buffer every layer

Every compressed layer runs `ttnn.fill_cache(scache.sel_tile, scache.kv, 0, ...)` to put its own
128-row ring over rows `[0, 128)` of the group's `sel_tile`, then attends the 640 rows (7.1 us per
layer). Attention over a key set is order-invariant: the keys are already rotated and SDPA's
softmax has no positional term. So the ring and the selection do not need to be adjacent rows of
one tensor.

* **Two-segment K for the MLA reader:** chunks `[0, window / k_chunk)` come from the layer's ring
  `scache.kv`, the rest from `sel_tile` at row `window`. This is a reader-only change in
  `sdpa_decode` (one more tensor accessor and a chunk-index branch). The causal bound stays
  `cur_pos = window + selected - 1`.
* Alternative without an op change: a paged view (block 32) whose page table lists the ring's 4
  blocks and the selection's 16. It needs ring and selection in one paged pool, a bigger cache
  layout change.

**Expected: -0.2 ms/token.** It also removes 30 writes from a shared buffer that today forces the
group's layers to run in order (the module docstring of `tt/decode/attention.py`).

### 3.5 Write the selection tiled

In the 6 index layers `fused_lightning_select_kv` writes `sel_kv` row-major and a `Tilize` (7.6 us)
converts it to `sel_tile`. Either the select gathers in groups of 32 rows and writes tiles (see the
indexer document), or 3.4's reader reads the selected segment row-major and tilizes it in L1, as
`tilize_q` does for Q. -0.05 ms/token, plus a DRAM round-trip.

## 4. Order of work and expected result

| step | change | risk | per token |
|---|---|---|---|
| 1 | `c_19` sized by tree rounds; sweep cores/head 4/8/16 | low (op-local, unit-tested) | -1.0 to -1.3 ms |
| 2 | output RoPE on 16 cores (3.3 A) | low | -0.6 ms |
| 3 | SDPA fidelity / exp-approx knobs, measured | accuracy check | -0.1 to -0.3 ms |
| 4 | two-segment K (3.4) | medium (op reader change) | -0.2 ms |
| 5 | RoPE in the SDPA reducer (3.3 B), tiled selection (3.5) | medium | -0.1 ms |

Together that is ~2.0-2.5 ms of the 27.8 ms token: **36 -> ~39-40 tok/s/u at 32k**. The SDPA share
does not grow with context: a compressed layer always attends 640 keys.

## 5. Validation

* Op: `tests/ttnn/unit_tests/operations/sdpa/test_mla_decode.py` (and the nightly
  `test_mla_decode_stress.py`) at each `max_cores_per_head_batch`, PCC against the torch reference.
* Block: `models/experimental/deepseek_v41_flash/tests/decode/test_attention.py`
  (`test_attention_decode`) for fidelity changes and the two-segment K.
* End to end: `test_full_model_decode_demo.py` at `DEEPSEEK_V4_START_POS=32768`, tok/s/u and text
  against the current build. Then re-profile 32L and diff `SdpaDecode`, `FusedPartialRope` (1-core
  calls) and `UpdateKVCacheOperation`.

## 6. Open questions

1. Does 8 cores per head fit with `keep_weights_in_l1` after 3.1, or does it need a stage-specific
   weight placement?
2. Is the tree round cost ~1-2 us as assumed? It sets where 16 cores per head stops paying.
3. Is HiFi2 accurate enough at `head_dim` 512 with one shared head? The softmax sums 640 terms,
   few enough that it should be.
