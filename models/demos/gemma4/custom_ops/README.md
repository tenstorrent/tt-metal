# Gemma 4 custom ops

Model-local fused kernels for Gemma 4 (`ttnn.generic_op` programs with their kernels beside them, the GLM 5.3
`custom_ops/` pattern). Each op replaces a chain of data-movement or small compute programs of the decode step or
the prefill with one program; the measurement protocol is `tests/unit/roofline/test_decode_roofline.py` (device
programs and device ms per step) plus the model gates. No code outside `models/demos/gemma4` carries a Gemma 4 kernel.

```
custom_ops/<op>/op.py        # class <Op>: golden(torch) + op(ttnn tensors) -> ttnn.generic_op(ProgramDescriptor)
custom_ops/<op>/kernels/     # reader / compute / writer (+ op-local headers)
custom_ops/kernels/          # headers shared by several ops (norm_rope_stages.hpp)
custom_ops/grid.py           # compute_grid (GEMMA4_TEST_GRID cap), the host-testable planners, the Blackhole L1 size
tests/unit/custom_ops/test_<op>.py
```

## Rules every op follows

* **Grid from the device.** A planner derives its work split from `grid.compute_grid(device)` (the device's
  `compute_with_storage_grid_size`, capped by `GEMMA4_TEST_GRID=WxH`), never from a literal; the planners are pure
  python with host tests over the 8x8 (Wormhole) and 13x10 / 12x10 / 11x10 / 10x10 (Blackhole P150 and the harvested
  P300) grids, and a device case forces the 8x8 split on Blackhole.
* **Cores per the workload.** The split uses the finest unit the kernels have and as many cores as the grid holds
  units (`min(units, grid)`): one output tile (`geglu_decode`, `moe_combine`, `concat_heads_decode`), one `[32, D]`
  head tile row (`qkv_norm_rope_prefill`), one (user, tensor) (`qkv_heads_norm_rope`). A lone row or a small batch
  is never pinned below that. Two deliberate exceptions, both measured: the fused norms' exchange group takes the
  FEWEST cores whose tiles fit one dest window (the all-to-all exchange grows with N^2: 11 cores 17.1 us vs 22
  cores 23.6 us vs 44 cores 41.3 us for one 2816-wide row tile on a P150), and the sharded GeGLU runs on the down
  matmul's activation grid (<= 8x8 by the DRAM-sharded matmul rule) so its output needs no reshard. See the audit
  table below for the counts at the served shapes.
* **Blackhole-only ops refuse through their wrapper.** `GegluRoutedExpert.unsupported_reason(device)` (the arch and
  the 11x8 compute rectangle its K-split, multicasts and padded N assume) and `MoeComputeCustom.unsupported_reason(...)`
  (the arch and the standalone staging capacity `E x T <= 64 x hs x ws` rows) are read by the expert wrappers
  (`tt/experts/__init__.py`, `tt/experts/moe_decode.py`): one warning, the composite path serves. The unit tests skip
  with the reason, never fail.
* **Contracts are asserted at the op, bounds at `supports()`.** A model shape the kernels cannot take (the CI's
  reduced-expert configurations, more local experts than the scratch regions hold, routing shards off the drain core)
  is refused by `supports()` / `unsupported_reason()` before a program is built; the kernels `static_assert` the same
  bounds on their compile-time args.
* **Trace-safe.** No host reads, fixed output shapes, semaphores reset by the kernels, buffer addresses as runtime
  args re-applied on a program-cache hit; a plan's cache key carries every input that changes the per-core argument
  layout (shapes, page sizes, memory configs).
* **Validation.** Every device test runs its op twice (the second on fresh buffers: the cache hit); the unit jobs run
  with the watcher on so the kernels' `ASSERT`s are compiled in, as the metal CI runs them; the host compile check
  `scratchpad/g2/kc/kcheck.sh` (TRISC / NCRISC command lines of a recorded build) runs before a device job. A
  header-only edit (`custom_ops/kernels/*.hpp`, the op-local `.hpp`) is picked up by the JIT's dependency hash.

## Ops and contracts

| op | program | contract | replaces |
|---|---|---|---|
| `fused_norm_decode` | the Gemma 4 RMSNorm family: `N` cores per row tile exchange 128-byte partial row sums, FPU column / row broadcasts, gamma as a row-replicated tile, one bf16 pack | `x [1, 1, R, W]` bf16 TILE interleaved (R <= 32 decode rows or tile-padded prefill rows), gammas `[1, 1, 32, W]` (`full_gamma`), forms: 1-3 outputs of one input, `+ residual`, `norm(x) g + norm(x2) g2`, `* scalar`; outputs in the input's config or a width-sharded decode matmul activation layout; `plan_norm_split` = `(N, per, groups, passes)` from the grid | per layer 8 norms + 3 adds + 1 mul (`to_memory_config, rms_norm, sharded_to_interleaved` each) -> 5 programs |
| `router_topk` | 32 RISC-V cores rank their user's row (top-k of the logits, the k exponentials against the row max, sum-normalized, times the per-expert scale); core 0 gathers | `logits [1, 1, B <= 32, E]` bf16 TILE, `pes [1, 1, 1, E]` bf16 ROW_MAJOR, `base [1, 1, 1, 16]` uint32 (this chip's first expert) -> `w [1, 1, E_loc, 32]` + `sparsity [1, 1, 1, E_loc]`; `op_packed` -> `[1, 1, 32, 2k]` (weights, global ids) with no gather; `supports(E, k, packed, e_loc)`: E a tile multiple <= 256, k <= 32, packed k % 8 == 0, e_loc even <= 64 | `softmax, topk, sum, div, zeros_like, scatter x5, mul, matmul, max, to_layout, permute` (15) / the packed tail (10) |
| `geglu_decode` | one output tile per work unit: exact GELU of the gate tile on fp32 dest times the up tile | `x [1, E, R, 2I]` bf16 TILE (`[up \| gate]`) -> `[1, E, R, I]` in the input's config | experts `transpose, slice, slice, gelu, mul` and the shared MLP's `slice, slice, gelu, mul` |
| `geglu_sharded` | the same compute on the down matmul's L1 width-sharded activation grid, operands fetched from the gate_up shard cores | `x [1, 1, R <= 32, 2I]` L1 WIDTH_SHARDED, `out_memory_config` = `DramShardedLinear.act_memory_config` | `sharded_to_interleaved, geglu, to_memory_config` (dense decode) |
| `moe_combine` | per output column tile the FPU column-broadcast products of the selected experts accumulated on fp32 dest through an L1 running sum | `down [1, E, R <= 32, H]` bf16 TILE, `w [1, 1, E, 32]` (router_topk's table) -> `[1, 1, R, H]`; reads only the selected experts' tiles | `permute, mul, fast_reduce_nc` |
| `qkv_norm_rope` | per user (core) on the height-sharded head shards bound as CBs: the norm stage, the FPU row-broadcast RoPE of the user's cos / sin row, in place | `q [1, B, Hq, D]`, `k / v [1, B, Hkv, D]` L1 HEIGHT_SHARDED (one user per core, shard `[32, D]`), `gq / gk [1, 1, 1, D]`, `cos / sin [1, 1, 32, D]` (row u = user u) -> the same tensors (`GEMMA4_QKV_NORM_ROPE_INPLACE=0`: fresh) | `to_memory_config x3, rms_norm x3, q/k RoPE (7 each), to_memory_config x2` (21) |
| `qkv_heads_norm_rope` | the head split, norms and RoPE straight from the L1 QKV projection output; one (user, tensor) per core when the grid holds 3B cores, else one user per core (`grid.plan_qkv_heads_units`) | `x [1, 1, B <= 32, (hq + 2 hkv) D]` bf16 TILE L1 interleaved, gammas, cos / sin rows -> `q [1, B, hq, D]`, `k / v [1, B, hkv, D]` L1 HEIGHT_SHARDED (the `nlp_create_qkv_heads_decode` layout) | `nlp_create_qkv_heads_decode` + the chain above (dense 31B / 12B) |
| `qkv_norm_rope_prefill` | work unit = one `[32, D]` tile row of one head (32 positions) dealt round-robin over the grid (`plan_units`), streamed through L1; elementwise RoPE with the cos / sin tile rows of the same positions; HiFi4 + fp32 dest | `q / k / v [B, H, S, D]` bf16 TILE (k / v None on KV-shared layers), gammas `[1, 1, 1, D]`, the `[1, 1, S_max, D]` cos / sin caches, `row_tile_offset` for a chunk -> fresh tensors in the inputs' configs | `rms_norm x3` with their reshape pairs and two `rotary_embedding` programs (11) |
| `concat_heads_decode` | work unit = (head-dim column tile, head group): the core reads the B input tiles of its column once and assembles the output tiles with 32-byte face-row copies (`grid.plan_concat_heads_units`) | SDPA output `[1, B, H (pad 32), D]` bf16 TILE DRAM -> o_proj input `[1, 1, B, H D]`; rows >= B zero | `to_memory_config, nlp_concat_heads_decode, sharded_to_interleaved (, slice)` |
| `geglu_routed_expert` | the unified routed-expert FFN with GeGLU on an 11x8 rectangle: per-K-block x and weight multicasts, writer-side NoC-1 up read and down split, runtime chunking from each expert's own count (`adaptive_chunk.hpp`), packer-L1 accumulation, LoFi; a partial tile-row's padding sticks are zero-filled and multicast | `x [M_buf, K]` bf16 ROW_MAJOR (tilized in-op) or bf8 TILE, `counts / region_offsets [G]` uint32 (per GLOBAL expert), `idx_table [E]`, `TtRoutedExpert`'s per-expert bf4 / bf8 weights (DRAM interleaved or ND-sharded one tile row tall) -> `[M_buf, K]` bf8 TILE at the expert regions (the padding rows of a region exactly zero), other rows untouched; `max_tokens` is the per-expert capacity (the kernels clamp and `ASSERT`) | `unified_routed_expert_ffn` (the shared op has no GeGLU) |
| `expert_gather_small` | one data-movement kernel on E_loc cores (one local expert each): a routing-column scan, a count exchange through the leader core (tile-aligned regions in local-expert order), the token rows copied into the GeGLU buffer, the combine weights `W [T, M]` written per column tile | `hidden [1, 1, T <= 256, H]` bf16 ROW_MAJOR and the dense `routing [1, 1, T, E]` replicated per chip, `idx [E_loc]` -> `buffer [M, H]`, `counts / offsets [1, E]` (zero for experts not here), `w [1, 1, T, M]`; `M = T K + 32 (E_loc - 1)` fixed at construction (drop-free); `supports(E, E_loc, K, H)`: E a tile multiple, E_loc <= 64 (the kernel's scratch regions), 2H <= 8192 | the token dispatch's dispatch and combine sides for a <= 256-token chunk (~22 programs); the combine is `ttnn.matmul(W, expert_out)` + one all-reduce |
| `moe_frame_prep` | B cores (one token each): the row-chip's frame rows untilized into the dispatch input, the packed routing split into uint16 ids + height-sharded scores, zero-weight padding rows synthesized | `hidden [1, 1, R, H]` bf16 TILE DRAM (lane-gathered frame), `packed [1, 1, R, 2k]` bf16 ROW_MAJOR, `rowbase [1, 1, 1, 16]` uint32 (`make_rowbase`) -> `x [B, 1, 1, H]` ROW_MAJOR in the dispatch config, `ids [B, 1, 1, k]` uint16, `scores [B, 1, 1, k]` bf16 (height-sharded) | `pad x2, to_layout x3, concat, mesh_partition x3, reshape x3, typecast, to_memory_config x3` (~15) |
| `sdpa_decode_splitfree` | paged decode attention in two programs sharing a persistent fp32 scratch: (1) every (row, KV group, 4096-token unit) on core `(unit mod M + row*G + group) mod n_cores` (a row's units on consecutive cores whatever the padded batch: a lone 256K row on 64 cores inside a B=32 graph as inside a B=1 graph) (`planner.py`, independent of `cur_pos`), the stock flash-decode recurrence over the unit's fixed 256-token chunks in ascending order with fp32 statistics, the causal / window masks as the stock kernels mask, the unit's (m, l, O) tiles to its scratch slot; (2) one core per (row, group) merging the live units in ascending order with the stock correction step, O / l once in bf16. A row's bits never depend on the padded batch or the cores (tt-metal#59300: the Galaxy pin `GEMMA4_SDPA_MAX_CORES_PER_HEAD=3` cost a lone 256K row 9.6x; chunk and unit are build constants); `GEMMA4_SDPA_SPLITFREE_UNIT=grid` puts one unit per core for a lone row (2048 tokens on 13x10) but measured slower on a 12x10 P150 (1.58 vs 1.31 ms at 256K, 1.11 vs 1.05 at 128K: the serial combine costs ~13 us per partial), so 4096 stays until the combine is a tree | `q [1, B <= 32, Hq <= 32, D]` bf16 TILE interleaved, paged `k / v [blocks, nkv, block (a tile multiple), D]` bf16 or bfp8, `page_table [B, max_blocks]` int32, `cur_pos [B]` int32 (-1 = padding), the stock geometry override (`num_kv_heads`, `block_size`), `sliding_window`, `cache_position_modulo`; `scratch` = `allocate_scratch(device, scratch_tiles(...))` once per model (B=32 x 64 units x 18 fp32 tiles per KV group = 144 MiB at 256K, head_dim 512; one slot per core under `=grid`); `unsupported_reason()` refuses out-of-contract calls (the stock op serves); HiFi2 + fp32 dest | `paged_scaled_dot_product_attention_decode` on the decode step (global layers at `GEMMA4_SDPA_DECODE_SPLITFREE=1`, sliding too at `2`) |
| `moe_compute` | the shared decode MoE kernel's ComputeOnly program (tilize cores -> DRAM-ring matmul cores -> dm1 staging) with every expert's W0/W1 and W2 slice resident in L1 for the call (one DRAM stream per expert whatever the chunk count), `layer_id` a runtime arg, zero-score routes dropped by the tilize scan (`skip_zero_scores`, on by default; `GEMMA4_MOE_COMPUTE_SKIP_ZERO_SCORES=0` keeps them) | `tilize_input [1, T, H]` bf16 ROW_MAJOR, `indices / scores [1, T, K]` L1 HEIGHT_SHARDED on the tilize drain core (asserted), `mapping [D, E_global]` uint16, the shared op's packed bf4 weights -> its five ComputeOnly outputs; `staging_layout` `"standalone"` (one block per expert, the layout `selective_reduce_combine` reads) or `"fused"` (the shared op's halves, for the bit-identity test); `TTMoEDecodeCustom` (`moe_decode_custom.py`) runs `TTMoEDecode.forward` with the one `moe_compute` call redirected to this op + the standalone combine | `ttnn.experimental.moe_compute` on the decode frames (the per-chunk weight re-stream) |
| `paged_update_rows` | the packed verify's K (or V) rows of every record row into the paged cache as one program: one (row, KV head) unit per core (`grid.plan_paged_update_rows`), per unit the <= 2 cache tile rows its P consecutive positions touch (through the ring modulo and the page table) as fixed jobs; per job the stock `paged_update_cache` read-modify-write (untilize the row's Wt tiles to bf16 row-major with the stock kernel_lib helper, patch the rows from the untilized source rows, tilize back in the cache format, store) by exactly one core, so no two cores touch one tile and the P rows go in one round trip (the stock's P sequential round trips are idempotent) | `src [1, nkv, S_pad, D]` bf16 TILE interleaved (the prefill-style K / V after norm and RoPE, row `b*P+p`), the loop's P `write_idxs [B]` int32 (-1 = skip), `page_table [B, blocks]` int32, the cache view (`num_kv_heads`, `block_size`), `cache_position_modulo`; a bf16 or bfp8 TILE DRAM cache updated in place; P <= 32, B <= 32; `unsupported_reason()` sends the rest to the loop | the per-position loop of `packed_decode_forward` (`packed_kv_write_loop`: slice, slice, reshard, reshard, update, update per position = 6 P programs per layer, 36 at K=5) -> 2 |
| `sdpa_decode_lse` | the stock paged flash decode (`paged_scaled_dot_product_attention_decode`'s three kernels copied, the factory's core split, K-chunk split and binary-tree merge re-expressed in `planner.py`) whose tree root also exports its final row statistics: `m` (row max of the raw scores) and `l` (sum of `exp((s - m) * scale)`), so `lse = scale * m + ln(l)`; O is the stock op's, bit for bit by construction | `q [1, B, Hq <= 32, D in {256, 512}]` bf16 TILE interleaved, paged `k / v [blocks, nkv, block, D]` bf16 or bfp8 with the stock view (`num_kv_heads` 1, `block_size` a tile multiple), `page_table [B, max_blocks]` int32, `cur_pos [B]` int32 (-1 = skip), causal, no mask / sink / window, the stock `SDPAProgramConfig` (k chunk > 0) and compute config -> `(out [1, B, Hq, D] bf16, stats [1, B, PNH, 64] bf16)`, `stats[..., h, 0] = m`, `stats[..., h, 32] = l`; `unsupported_reason()` refuses the rest | the global layers' decode attention under the pair split (`tt/sdpa_pair_split.py`) and the prefix call of the serving verify (`tt/verify_prefix_lse.py`); `pair_split=True` attends one half of each row's key chunks per chip of a KV-head pair |

## Defaults per build

The model build records its profile once (`tt/mesh_orientation.py`: `decode_fusion_defaults`, read through
`decode_flag`); a set env var always wins (`0` restores the composite path).

| switch | op | 26B (MoE) on 4 Blackhole chips, (1,4) or (4,1) | dense 31B / 12B on 4 Blackhole chips | Galaxy lanes (8,4), one chip, Wormhole |
|---|---|---|---|---|
| `GEMMA4_NORM_FUSED=2` + `GEMMA4_LAYER_NORM_FUSED=1` | `fused_norm_decode` (single norms + the five-program layer tail) | on | off (the native width-sharded LayerNorm is faster on the dense widths) | off |
| `GEMMA4_MOE_FUSED=1` | `router_topk`, `geglu_decode` (experts), `moe_combine`; `router_topk.op_packed` + `moe_frame_prep` on the decode MoE kernel rows | on | - | off |
| `GEMMA4_MLP_GEGLU_FUSED=1` | `geglu_decode` (shared MLP, decode rows) | on | off | off |
| `GEMMA4_MLP_GEGLU_SHARDED=1` | `geglu_sharded` | off | on | off |
| `GEMMA4_QKV_NORM_ROPE_FUSED=1` | `qkv_norm_rope` | on | off | off |
| `GEMMA4_QKV_HEADS_FUSED=1` | `qkv_heads_norm_rope` | off | on | off |
| `GEMMA4_CONCAT_HEADS_FUSED=1` | `concat_heads_decode` | on | on (`GEMMA4_CONCAT_HEADS_SHARDED=1`) | off |
| `GEMMA4_QKV_NORM_ROPE_PREFILL=1` | `qkv_norm_rope_prefill` | off | on (Blackhole only; refused on Wormhole) | off |
| `GEMMA4_SMALL_ROW_EXPERTS=1` (32..128-token calls) | `expert_gather_small` + `geglu_routed_expert` + `W @ expert_out` | on with the dispatch prefill ((4,1) column mesh) | - | off |
| dispatch prefill (`GEMMA4_MOE_DISPATCH_PREFILL`) | `moe_dispatch_prefill` + `geglu_routed_expert` | on, (4,1) column mesh | - | off (Blackhole only: refused with one warning elsewhere) |
| dispatch prefill (`GEMMA4_MOE_DISPATCH_PREFILL`) | `geglu_routed_expert` | on, (4,1) column mesh | - | off (Blackhole only: refused with one warning elsewhere) |
| `GEMMA4_SDPA_DECODE_SPLITFREE=1` (`2`: sliding layers too) | `sdpa_decode_splitfree` | off (device validation pending; `tests/test_sdpa_decode_splitfree.py`) | off | off |
| `GEMMA4_MOE_DECODE_KERNEL=1` + `GEMMA4_MOE_COMPUTE_CUSTOM_OP=1` | `moe_compute` (decode frames; prefill rows keep the shared op) | off | - | opt-in on the Galaxy lanes (refused with one warning where the staging does not fit, e.g. a (2,4) block at frame 128) |
| `GEMMA4_PACKED_KV_WRITE_OP=1` | `paged_update_rows` (the packed verify's KV write of the speculative records) | off (device A/B pending; `tests/unit/custom_ops/test_paged_update_rows.py`) | off | off |

Test knobs of `sdpa_decode_splitfree`: `GEMMA4_SDPA_SPLITFREE_CHUNK` (the K chunk cap, default 256: the largest
rung of `planner.CHUNK_LADDER` that fits the arch's L1 serves), `GEMMA4_SDPA_SPLITFREE_UNIT` (the unit, default
4096; `grid` derives one unit per core, `planner.derive_unit`; a sliding layer's unit is one chunk), `GEMMA4_SDPA_SPLITFREE_READS` (K/V tile reads in flight per core,
default 16). The chunk and the unit are part of the row's bits: the same values on every decode graph of a build.

Call-site switches without a kernel: `GEMMA4_QKV_L1_OUT`, `GEMMA4_SOFTCAP_FUSED`, `GEMMA4_HEAD_NORM_4D`, the 1D decode
matmul configs (`GEMMA4_MLP/ATTN/ROUTER_DECODE_1D`), `GEMMA4_NORM_SHARDED_OUT`, `GEMMA4_DRAM_SHARD_ROW_VIEW`,
`GEMMA4_MOE_DECODE_TILE_TAIL`. Test knobs: `GEMMA4_TEST_GRID=WxH` (cap the grid), `GEMMA4_FUSED_NORM_CORES` (a
preferred exchange-group size for the fused norms, taken when it fits; unset: the fewest cores that fit one dest
window, 11 for the 26B), `GEMMA4_QKV_NORM_ROPE_INPLACE`, `GEMMA4_CONCAT_HEADS_PER_UNIT`.

## Cores at the served shapes (audit 2026-10-30)

Per chip, from the planners (`tests/unit/test_custom_op_planners_host.py` asserts the counts); BH = the 13x10 P150
grid (130 cores), WH = the 8x8 Wormhole grid (64). "Units" is the finest split the kernels have; an op uses
`min(units, grid)` cores unless noted. The dense 31B / 12B TP4 and the 26B (4,1) builds on 4 Blackhole chips are the
served QB2 shapes; the Galaxy (8,4) engines run the dense fusions off (only `qkv_norm_rope` on the 26B lanes, and the
split-independent decode SDPA of the `arg/gemma4_sdpa_splitfree` branch where enabled).

| op | served shape (per chip) | units | BH cores | WH cores | rule / note |
|---|---|---|---|---|---|
| `concat_heads_decode` | 26B / 12B TP4, 4 heads x D 256 / 512 | 32 / 64 | 32 / 64 | 32 / 64 | one output tile per unit |
| | 31B TP4, 8 heads x D 256 / 512 | 64 / 128 | 64 / 128 | 64 / 64 (head pairs) | |
| `qkv_heads_norm_rope` | dense TP4, B users | 3 B | 3 B (B 1: 3, 16: 48, 24: 72) | 3 B up to 21, then B | one (user, tensor) per core; a user's heads share one tile |
| `qkv_norm_rope` | 26B, B users (the head shards) | B | B (1 / 20 / 32) | B | in place on the shard cores |
| `fused_norm_decode` | 26B W 2816, one row tile | 11 | 11 (22 two inputs) | 11 (22) | fewest cores that fit a dest window (measured, see above) |
| | 26B 128 / 2048 prefill rows | 4 / 64 row tiles | 44 (1 pass) / 121 (6 passes) | 44 (1) / 55 (13) | one 11-core group per row tile |
| `geglu_decode` | 26B experts 32 x I 704 / shared I 2112 | 704 / 66 | 130 / 66 | 64 / 64 | one output tile per unit |
| `geglu_sharded` | 31B / 12B TP4 down activation | shard cores | 56 / 40 | 56 / 40 | the DRAM-sharded down matmul's grid |
| `moe_combine` | 26B H 2816 | 88 | 88 | 64 | one output column tile per unit |
| `qkv_norm_rope_prefill` | 31B TP4 S 128 / 2048 (16 heads) | 64 / 1024 | 64 / 130 | 64 / 64 | one `[32, D]` head tile row per unit |
| | 12B TP4 S 128 (8 heads / q-only 4) | 32 / 16 | 32 / 16 | 32 / 16 | |
| `router_topk` | 26B E 128, k 8 | 32 users | 32 | 32 | one user row per RISC-V core; scalar top-k |
| `moe_frame_prep` | Galaxy frame 32 / (2,4) frame 64 tokens | B | B | B | one token row per core (5.6 KB each) |
| `expert_gather_small` | 26B (4,1), 32 local experts | 32 | 32 | 32 | one local expert per core (the count exchange) |
| `geglu_routed_expert` | 26B (4,1), N 704 = 22 tiles | 11x8 rectangle | 88 | refused (Blackhole-only) | the shared op's rectangle: 22 N tiles pad-free over 11 columns, DRAM-bound (~450 GB/s) |
| `moe_compute` | 26B Galaxy lanes (opt-in) | the shared op's placement | 8 ring + combine + <= 4 tilize | refused | the DRAM-bank ring of `select_moe_compute_cores` |
| `moe_dispatch_prefill` | 26B (4,1), 1 link | per link | 1 sender + 2 workers / + <= 4 untilizers | same | fabric-link bound (the shared op's core lines) |

## `sdpa_decode_lse`: decode attention with the row LSE (the pair-split building block)

At long context a global layer's decode attention is DRAM-bound (about 370 GB/s; 1.57 ms per layer at 4 users x
131K on one Blackhole chip). The two chips that hold the same KV head can each attend half of a row's keys and
merge: `O = (e^(lse_a - M) O_a + e^(lse_b - M) O_b) / (e^(lse_a - M) + e^(lse_b - M))`, `M = max(lse_a, lse_b)`,
halving each chip's bytes. Chip B's half is the same call through the page table shifted by `mid / block` blocks
with `cur_pos - mid` (`mid` a block multiple); a row shorter than `mid` skips chip B (`cur_pos -1`, weight 0).

* **Kernels**: `kernels/{reader,writer}_decode_lse.cpp` and `sdpa_flash_decode_lse.cpp` are copies of the stock
  `reader_decode_all.cpp`, `writer_decode_all.cpp` and `sdpa_flash_decode.cpp` (tt-metal 298c06426a8); every edit is
  marked `[sdpa_decode_lse]`. The root's compute copies `cb_prev_max` / `cb_prev_sum` (without popping) into
  `cb_out_m` / `cb_out_l` -- the CBs a non-root core sends its partial through, unused on the root -- before the
  `1 / l` normalization; the root's writer stores them (writer runtime arg 1 = the stats address, the stats
  TensorAccessorArgs after the output's). The shared headers (`dataflow_common.hpp`, `rt_args_common.hpp`,
  `compute_common.hpp`) are included from ttnn.
* **Why (m, l) and not one fp32 LSE**: the stock keeps the statistics in bf16 CBs and the model's call runs the op
  default (HiFi2, fp32 dest OFF), so an LSE formed on device would be a bf16 number (0.125 ulp at |lse| ~ 30). `m`
  is exact (a max of bf16 scores), `l` has bf16's 2^-9 relative error and is the very `l` O was normalised by. Form
  `lse` (or the merge weights) in fp32 from the two (`SdpaDecodeLse.lse_from_stats`, `merge_torch`).
* **Dropped stock features** (refused): attention mask, sink, sliding window, MLA, GQA (> 1 KV head per chip),
  sharded / row-major Q, sharded output / page table / cur_pos, sub-core grids, the dynamic K chunk (`k_chunk_size
  0`), sub-tile blocks, the speculative multi-position mode. Only the active cores carry the program, and the
  tree-merge staging CB holds one block per round instead of `cores_per_head - 1` (same values).
* **Tests**: host `tests/unit/test_custom_op_planners_host.py -k sdpa_decode_lse` (the tree against a flat
  merge for every group size <= 64, the factory's split, argument layouts, the L1 fit); device
  `GEMMA4_SDPA_LSE_TEST=1 pytest tests/unit/custom_ops/test_sdpa_decode_lse.py` (O bitwise against the stock op,
  PCC and LSE against torch, the two-half merge against one full call, the cache repeat, the refusals).

## Kernel traps (Blackhole, `ttnn.generic_op`)

* An FPU write (`mul_tiles*`, `copy_tile`) to a dest tile that an earlier SFPU op of the same acquire window reads
  corrupts the result (`moe_combine`: wrong sums from 3 experts on). Issue every unpack copy / FPU op of a window
  before its first SFPU op, or pass the running value through an L1 CB between windows.
* `pack_tile(idst, cb, index)` writes sequentially from the reserved block regardless of `index`; an out-of-order
  index needs `pack_tile<true>(...)` (`qkv_norm_rope`: pages 0,4,1,5 landed in 0,1,2,3). Pack in page order.
* The math core (TRISC1) has no `cb_interface`: a compute kernel cannot read a data-dependent count from a CB
  (`moe_combine`: fixed expert loop, the reader pushes zero weight tiles for the experts nobody selected).
* `get_read_ptr` is a dataflow-only free function; compute kernels use `CircularBuffer::get_read_ptr()`.
* A kernel-side `union F32` collides with `dprint.h`; the ROW_MAJOR / TILE row tile of the position-gathered
  cos / sin can be copied per user by the reader (32-byte face rows), so no per-step transpose / slice / repeat.
* Every L1 staging region of a DRAM transfer is 64-byte aligned (a 32-byte-aligned stick read lands shifted on
  Blackhole; `expert_gather_small` static_asserts its scratch layout).
* A CB wraps only when a push lands exactly on its limit: every ring is a whole number of the finest granule its
  kernels push, and a per-expert block size would run the pointer away (`geglu_routed_expert/kernels/adaptive_chunk.hpp`).
* A tile-row the writer emits must hold finite values in every row: stale L1 in the padding sticks of a partial
  tile-row reached the output and `0 x NaN` poisoned the small-row combine (`geglu_routed_expert` zero-fills them).
* The first device case of a kernel with cross-core synchronization runs alone in its own short job, single-core ->
  2-core -> 4-core -> full grid.
