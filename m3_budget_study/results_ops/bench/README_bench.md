# M3 prefill op microbenchmarks, (2,4) sub-mesh

Each script is one standalone Python process. It opens the 8x4 Blackhole galaxy with `FABRIC_1D`, takes
`create_submeshes(MeshShape(2,4))[0]` (SP=2 on axis 0, TP=4 on axis 1, EP=8, so 16 routed experts per chip),
runs one grid, writes one CSV row per grid point and closes the mesh. `--dry-run` prints the grid and the
analytic FLOP/byte counts. It does not import ttnn.

| script | CSV | what |
|---|---|---|
| `bench_experts.py` | `experts.csv` | `TtRoutedExpert.forward`. `nd` = `unified_routed_expert_moe` only; `hybrid` = threshold 128 (`moe_fused_swiglu` for counts <= 128, the composite for the rest). Both use bf4 ND-sharded weights and SwiGluOai. |
| `fit_experts.py` | `experts_fit.txt` | Fits `t = a*W + b*T` and `t = max(a*W, b*T)`, plus `+c` variants, per path. |
| `bench_moe_reduce.py` | `moe_reduce.csv` | `post_combine_reduce`, the fused top-k weighted sum of `TtMiniMaxReduce`. The TP reduce-scatter is excluded. |
| `bench_msa.py` | `msa.csv` | `indexer_score_msa`, `topk_large_indices`, `sparse_sdpa_msa`, the full `msa_indexer_sparse` chain and `index_branch_forward`, all at cache-read shapes. |
| `run_bench.sh` | `../logs/` | Checks the lock, runs `tt-smi -glx_reset`, then runs the script under a 20-minute timeout. |

## Timing

The default is `BENCH_TIMING=device`: the device profiler with no tracy capture. `bench_common.setup_env()` sets
these before ttnn is imported:

- `TT_METAL_DEVICE_PROFILER=1`
- `TT_METAL_PROFILER_MID_RUN_DUMP=1`
- `TT_METAL_PROFILER_CPP_POST_PROCESS=1`
- `TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1`

Each timed call is followed by `synchronize_device`, then `ttnn.ReadDeviceProfiler(submesh)`, then
`ttnn.get_latest_programs_perf_data()`. The call's time on a chip is the sum of `DEVICE KERNEL DURATION [ns]` over
the programs that read added for that chip. Programs are de-duplicated by `program_execution_uid`, and only the
sub-mesh's chip ids are kept.

The run does 2 warm-up calls, then `--repeats` (default 5) timed calls. Each chip gets the median over repeats.
The CSV then reports `worst_ms` / `mean_ms` / `min_ms` across the 8 chips, along with `per_chip_ms`, `n_programs`,
`cores_used` (the profiler's `core_count`, max over the window's programs) and `wall_ms`. `wall_ms` includes the
profiler read, so it is only a sanity number.

If the API returns nothing, or `BENCH_TIMING=wall` is set, timing falls back to wall time around
`synchronize_device`. `worst`/`mean`/`min` are then taken over repeats, and `timing_src=wall`.

## Inputs

- **experts**
  - Weights: one random bf4 host tensor per projection shape, uploaded 16 times to separate ND-sharded DRAM buffers:
    gate/up `[6144,3072]`, down `[3072,6144]`, placed with `routed_expert_weight_memory_config`. The upload is swapped
    into `TtRoutedExpert` in place of its 128-expert conversion. The forward and the op arguments are M3's.
  - Buffer: `[16864, 6144]` bf16 ROW_MAJOR dispatch buffer. That is `compute_constants` at seq/chip 2048 with
    factor = topk, and gives `max_tokens` = 4096.
  - Routing: the first `active` local experts of each chip get `tokens_per_expert` rows. Regions are tile-aligned
    and laid out like `offset_cumsum`. The (4,2,128) counts and offsets tables are sharded with
    `shard_expert_token_counts`, and the global-expert table comes from M3's `get_ep_mesh_mapper`.
  - `W` = active x 3 x 6144 x 3072 x 0.5625 B. The ND-shard N padding (per_core_N x 11 tiles) is not counted.
  - `flops` = 2 x T x 3 x 6144 x 3072.
- **moe_reduce**
  - combine `[1,1,t,4,6144]` bf16 RM, weights `[1,t,4]` bf16 unsqueezed as in the module, and indices `[1,t,4]`
    uint16 holding 4 distinct random experts. All are replicated to every chip.
  - Dispatch table: M3's `create_dispatch_table(128,2,4)`, sharded per mesh column, so about 1 of 4 slots is local.
  - `bytes_moved` = t·4·6144·2 (the reader reads every slot) + t·6144·2 (output) + weights + indices.
- **msa**, per chip
  - q `[1,16,rows,128]` bf16 and index_q `[1,1,rows,128]` bf16.
  - K, V and index_k `[1,1,T,128]` bf8. With `M3_INDEX_CACHE_BF16=1`, index_k is bf16.
  - T = seq_local x 2. That is the persistent gather buffer, with rank r at slot r·seq_local.
  - `chunk_local` = rows, `cached_len` = kv_len − 2·rows, `chunk_start_idx` = cached_len.
  - `cluster_axis` = `block_cyclic_sp_axis` = 0, `block_cyclic_chunk_local` = rows, `kv_len` bound,
    `num_groups` = 1. The indexer uses `IndexerScoreProgramConfig(64,1024,0)`.
  - Points with kv_len < 2·rows (4096 at rows 4096) are written as SKIP.
  - index_branch: bf8 `[6144,128]` projections, unit norm gains, and indexed RoPE on random whole-cache cos/sin
    `[1,1,T,64]` sharded along SP.

**Needs a device to confirm:**
- ND-sharded `to_device` of a replicated bf4 host tensor.
- `ReadDeviceProfiler` on a sub-mesh, and whether "latest" data is refreshed per chip.
- Whether the MSA kernels accept these exact T/kv_len/chunk_local combinations. They follow `msa_cache_read_extent`,
  but only the `(8,4)` unit test exercises them.

## Op structure (from the program factories)

### post_combine_reduce (`experimental/deepseek_prefill/post_combine_reduce`)

**Grid:** `num_cores = min(tokens/32, grid.x*grid.y)` on `compute_with_storage_grid_size()`, row-major. That is 32,
64 and 128 chunks for t = 1024, 2048 and 4096. For t = 4096 the grid caps it and some cores get 2 chunks. Each core
takes whole 32-token chunks, and the first `chunks % cores` cores take one extra. emb is never split.

**Per-core loop, `for chunk: for token(32): for slot(4)`:**
- **Reader:** reads one bf16 emb row (12 KB) per slot, with no skipping.
- **Writer:** loads the dispatch table once and 32 index pages per chunk. Per slot it reads one weight page. If the
  token has no local expert, it zeroes the last slot's weight.
- **Compute:** skips slots whose `table[idx] == -1`. The `must_zero_init` slot (last slot, when no earlier slot was
  local) is forced through with weight 0. Each processed slot does a `mul_tiles_bcast` SCALAR and accumulates in
  the packer's L1 into a row-major scratch buffer, and the 32 rows are tilized into the TILE output.
- **CBs:** c_0 holds ceil(emb/1024) = 6 tiles; c_16 and c_17 hold 32x6 tiles. emb must be ≤ 8192 because a whole row
  is one DEST batch.

Traffic is set by the reader (4 full slots per token), not by compute (about 1 slot).

### indexer_score_msa (`experimental/indexer_score`, the DSA kernel with a synthesized gate)

**Grid:** a banded rectangle on `compute_with_storage_grid_size()`.
- There are G = Sq_tiles/QC q-row-groups, with QC = 64/32 = 2 tiles, so G = rows/64.
- `rows_for_groups` takes the largest divisor of G that is ≤ grid_y. Q is multicast along a grid row.
- There are U = ceil(kv_len_tiles/KC) k-bands, with KC = 1024/32 = 32 tiles. They map to `min(U, grid_x)` columns,
  and K is multicast down a column.
- If the groups under-fill the rows, `band_row_blocks` replicates groups across idle rows. Extra groups are
  phase-stacked.

**Parallel unit:** a (QC q-tiles x KC k-tiles) cell. Heads are not split; at TP=4 there is 1 index head.

**Per-core loop, `for phase (q-group): for band`:**
1. Q stays resident; the core waits for the K band.
2. QK matmul.
3. The causal suffix is masked to −inf. Every group computes the full `[0, kv_len)` rectangle and masks in-band,
   so FLOPs ≈ 2·rows·kv_len·128 with no causal saving.
4. Constant gate, reduce over the group's heads, 128-block max-pool, then untilize to bf16 RM
   `[1,1,rows,T/128]`.

Only math fidelity is honoured (LoFi for bf8 x bf8).

### topk_large_indices (`experimental/topk_large_indices`)

**Grid:** every worker core of the sub-device, or `sub_core_grids`. Rows (the product of the leading dims, so
`rows` here) are split with `split_work_to_cores`.

**Per-core loop:** for each row, it streams LLK windows of 512/1024/2048 elements (k is snapped). It sorts the
first window, then sorts and merges each later window into the survivor set. The mode is chosen at compile time:
Classic, FusedEndToEnd (≤ 32 windows) or FusedSegmented.

The row length is T/128 blocks (32 to 4288 here), bounded by `valid_length` = kv_len/128. The output is uint32
`[.., 16]`, and a surviving −inf becomes 0xFFFFFFFF, which is the −1 tail for sparse_sdpa. Compute uses
fp32 DEST and full sync.

### sparse_sdpa_msa (`transformer/sdpa/device/sparse_sdpa_msa_*`)

**Grid:** always the full `compute_with_storage_grid_size()`, with no program config. The work is `S·n_kv`
(query token, kv group) items, rows·1 here, split evenly with the remainder going to the first cores.

**Per work item:**
1. The reader loads the 16 q-head rows of the group (padded to one 32-row tile) and the token's 16 block ids,
   finding the −1 tail to get `n_active`.
2. For each active block, the reader fetches the upper half of the K/V tiles and the writer fetches the lower half.
   Block-cyclic remap is done in-kernel. Compute runs QK, then the causal mask on the diagonal block, online
   softmax, and PV.
3. Normalize, untilize, and write the row-major output.

**Cost:** K/V are re-fetched per query token, so traffic ≈ rows·16·128·128·2·1.06 B. At rows = 4096 that is
about 2.3 GB, independent of kv_len.

## Launch

```
cd /home/vmelnykov/tt-metal
m3_budget_study/results_ops/bench/run_bench.sh experts      # then fit_experts.py runs automatically
m3_budget_study/results_ops/bench/run_bench.sh moe_reduce
m3_budget_study/results_ops/bench/run_bench.sh msa
m3_budget_study/results_ops/bench/run_bench.sh all          # the three in sequence, reset before each
```

Extra arguments after the name go to the script, for example `run_bench.sh msa --ops indexer,topk`. The launcher
also honours `BENCH_TIMEOUT` (seconds, default 1200) and `BENCH_TIMING=wall`.
