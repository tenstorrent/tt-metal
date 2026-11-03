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
| `moe_compute` | the shared decode MoE kernel's ComputeOnly program (tilize cores -> DRAM-ring matmul cores -> dm1 staging) with every expert's W0/W1 and W2 slice resident in L1 for the call (one DRAM stream per expert whatever the chunk count), `layer_id` a runtime arg, zero-score routes dropped by the tilize scan (`skip_zero_scores`, on by default; `GEMMA4_MOE_COMPUTE_SKIP_ZERO_SCORES=0` keeps them) | `tilize_input [1, T, H]` bf16 ROW_MAJOR, `indices / scores [1, T, K]` L1 HEIGHT_SHARDED on the tilize drain core (asserted), `mapping [D, E_global]` uint16, the shared op's packed bf4 weights -> its five ComputeOnly outputs; `staging_layout` `"standalone"` (one block per expert, the layout `selective_reduce_combine` reads) or `"fused"` (the shared op's halves, for the bit-identity test); `TTMoEDecodeCustom` (`moe_decode_custom.py`) runs `TTMoEDecode.forward` with the one `moe_compute` call redirected to this op + the standalone combine | `ttnn.experimental.moe_compute` on the decode frames (the per-chunk weight re-stream) |

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
| dispatch prefill (`GEMMA4_MOE_DISPATCH_PREFILL`) | `geglu_routed_expert` | on, (4,1) column mesh | - | off (Blackhole only: refused with one warning elsewhere) |
| `GEMMA4_MOE_DECODE_KERNEL=1` + `GEMMA4_MOE_COMPUTE_CUSTOM_OP=1` | `moe_compute` (decode frames; prefill rows keep the shared op) | off | - | opt-in on the Galaxy lanes (refused with one warning where the staging does not fit, e.g. a (2,4) block at frame 128) |

Call-site switches without a kernel: `GEMMA4_QKV_L1_OUT`, `GEMMA4_SOFTCAP_FUSED`, `GEMMA4_HEAD_NORM_4D`, the 1D decode
matmul configs (`GEMMA4_MLP/ATTN/ROUTER_DECODE_1D`), `GEMMA4_NORM_SHARDED_OUT`, `GEMMA4_DRAM_SHARD_ROW_VIEW`,
`GEMMA4_MOE_DECODE_TILE_TAIL`. Test knobs: `GEMMA4_TEST_GRID=WxH` (cap the grid), `GEMMA4_FUSED_NORM_CORES` (the fused
norms' preferred group size, default 22), `GEMMA4_QKV_NORM_ROPE_INPLACE`, `GEMMA4_CONCAT_HEADS_PER_UNIT`.

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
