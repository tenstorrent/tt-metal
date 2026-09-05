# Existing fast paths

**Read this before proposing any new kernel or fusion.** ttnn already carries a
large set of DiT-specific fused ops — several written for this exact codebase —
and a hot spot in your profile is far more likely to be a fast path that isn't
engaged than one that doesn't exist.

Two questions when you find a hot spot:

1. **Is there a fused op for this pattern?** Tables below.
2. **If a fast path exists but the profile shows the unfused ops, why?** Almost
   always a shape guard, a dtype mismatch, a layout mismatch, or a call site that
   predates the op. That is a config fix, not kernel work.

**Scope: ops usable from a DiT forward pass.** Ops whose constraints exclude DiT
shapes are deliberately absent — e.g. `rotary_embedding_llama_fused_qk` is
decode-only (`seq_len=1`) and `convert_to_chw` caps channels at a tile height, so
neither applies here. If an op you expect is missing, that is usually why.

Verify against the source tree, not a locally built ttnn — a branch build can
carry ops that are not on `main`:
`grep -rl "<op>" ttnn/cpp/ttnn/operations/` then
`./python_env/bin/python -c "import ttnn; help(ttnn.experimental.<op>)"`.

---

## Normalization + collective (DiT-specific)

These were built for sharded diffusion models. If you are writing distributed
normalization by hand, stop.

| Op | Does | Used by |
|---|---|---|
| `ttnn.experimental.dit_fused_distributed_rmsnorm` (+ `_create_stats_buffer`) | Distributed RMSNorm, stats gathered across the mesh | `layers/normalization.py` |
| `ttnn.experimental.dit_fused_distributed_layernorm` (+ `_create_stats_buffer`) | Distributed LayerNorm, same shape of contract | `layers/normalization.py` |
| `ttnn.experimental.dit_layernorm_pre_allgather` / `dit_layernorm_post_allgather` | The two halves separately, when you need to place the all-gather yourself | `tests/unit/test_distributed_rmsnorm_fused.py` |
| `ttnn.experimental.dit_rms_norm_unary_fused` | **RMSNorm + activation in one kernel pass.** Equivalent to `ttnn.silu(ttnn.rms_norm(x))` / `gelu`, without the intermediate write+read | `layers/normalization.py`, `encoders/gemma/` |

**GroupNorm has no fused distributed variant** — RMSNorm and LayerNorm do, which
makes the gap surprising. `layers/normalization.py::GroupNorm3D` drives the local
`ttnn.group_norm` with the grid pinned by
`determine_expected_group_norm_dram_grid_size`; cross-device statistics are yours
to assemble.

**Workflow rule.** A profile showing `RMSNorm` immediately followed by `Silu` or
`Gelu` means `dit_rms_norm_unary_fused` is not engaged. A profile showing
`AllGather` between two norm halves means you hand-rolled what
`dit_fused_distributed_*` does in one dispatch.

### The norm is a fusion hub — fold everything into it

`layers/normalization.py::DistributedRMSNorm.forward` collapses **four**
operations into one device op. Every one of these is an elementwise or
data-movement op you would otherwise pay for per layer per step.

| Fold in via | Replaces | Note |
|---|---|---|
| `dynamic_weight=` | A separate AdaLN scale multiply | The effective weight becomes `static_weight × dynamic_weight`, applied inside the fused op at fp32. RMSNorm has no bias term, so scale only |
| `num_heads_per_device=` | A reshape into heads after the norm | One RMSNorm over the full per-device row, then head split, in-op |
| `per_head_norm=True` | Per-head QK-norm as a separate op | RMSNorm **independently per head** over head_dim; no cross-device all-gather since each head is device-local. **Not equivalent** to whole-row — models must opt in explicitly |
| `rope_cos=`, `rope_sin=`, `transformation_mat=` | A standalone rotary-embedding dispatch | Norm + head split + RoPE in a single pass |

So the fully-fused form is **norm + AdaLN scale + head split + RoPE**, one op
where a naive port emits four.

`DistributedLayerNorm.forward` absorbs AdaLN scale **and** shift, since LayerNorm
has a bias term — `dynamic_weight=(1.0 + scale)`, `dynamic_bias=shift`. It
asserts they are supplied together. This is the Wan block pattern (`transformer_wan.py`, the `self.norm1(...)` call
in the block forward) and LTX reuses it for `norm_out` in `transformer_ltx.py`,
commented "Fuse the AdaLN (1 + scale) * normed + shift modulation into norm_out
(WAN pattern)".

Buffer sizing depends on the combination, so `get_fused_norm_stats_buffer` is
keyed on `(shape, heads-per-device, per_head_norm, rope present, weight shape)`
— a shared-cache collision between two same-shape modules differing only in
affine geometry is a real hazard the key exists to prevent.

### Other DiT-shaped fusions

| Pattern | Fused form | Where |
|---|---|---|
| **AdaLN coefficients laid out on an outer dim** | `Parameter(total_shape=[coeff, 1, 1, D])`, then `ttnn.chunk(x, coeff, dim=0)` | Keeps each modulation parameter on the **non-tiled** dim 0, so the per-block chunk is a free tile-aligned slice — avoids `untilize → slice → re-tilize`. See `adaln_coeff` in `transformer_ltx.py` (6, or 9 with cross-attention AdaLN) |
| **Fused column-parallel QKV** | `ColParallelLinear(dim, 3 * dim, chunks=3)` | One GEMM producing three chunks (Q, K, V) instead of three matmuls. See `self.to_qkv` in `attention_ltx.py`. `minimal_matmul_split` is the split-output primitive |
| **Output-projection epilogue with gate** | `dit_minimal_matmul_addcmul_fused` | `residual + gate × (matmul + bias)` in one dispatch — the whole block epilogue |

---

## Matmul and matmul+collective

| Op | Does | Used by |
|---|---|---|
| `ttnn.experimental.minimal_matmul` | High-performance `A @ B [+ bias]`, TILE layout, tile units internally. **Takes `fused_activation=` and `fuse_swiglu=`** | `layers/linear.py` and attention throughout |
| `ttnn.experimental.minimal_matmul_split` | Split-output variant | `layers/linear.py` |
| `ttnn.experimental.all_gather_minimal_matmul_async` | **All-gather fused into the matmul**, same `fused_activation` / `fuse_swiglu` surface. This is the TP linear you want, not AG-then-matmul | `layers/linear.py`, `models/transformers/ltx/attention_ltx.py`, `wan2_2/attention_wan.py` |
| `ttnn.experimental.minimal_matmul_strided_reduce_scatter_async` | Matmul fused with strided reduce-scatter — the row-parallel counterpart | `layers/linear.py` |
| `ttnn.experimental.dit_minimal_matmul_addcmul_fused` | **AdaLN modulation in one dispatch:** `out = in1 + (scalar * matmul(x, W) * in2)`, addcmul computed inline in the matmul kernels | `models/transformers/ltx/attention_ltx.py`, `wan2_2/attention_wan.py` |
| `ttnn.experimental.strided_all_gather_minimal_matmul_async` | Strided AG + matmul | *bound, no tt_dit caller yet* |
| `ttnn.experimental.matmul_reduce_scatter_async` | Generic matmul + RS | *bound, no tt_dit caller yet* |

**Workflow rule.** `AllGather` or `ReduceScatter` adjacent to a `Matmul` in the
top ops means the fused variant isn't engaged. Likewise a `BinaryNg` chain right
after a matmul in an AdaLN block — that is
`dit_minimal_matmul_addcmul_fused`'s exact pattern.

**Known-good compute config** for a fused linear, from `layers/linear.py`:
HiFi2 + `packer_l1_acc=True` + `fp32_dest_acc_en=True` + `math_approx_mode=False`.
Its comment calls this "the special config which attains good correctness" —
start there rather than re-deriving.

---

## Attention

| Op | Does | Used by |
|---|---|---|
| `ttnn.transformer.scaled_dot_product_attention` | Base SDPA. **Chunk size is the tunable** — sweep at the real shape (`../tt-dit-performance/optimization-levers.md`) | Throughout |
| `ttnn.transformer.ring_joint_scaled_dot_product_attention` | **Ring attention for sequence parallel, joint text+image streams.** The SP attention for DiT | LTX, Wan, SD3.5 |
| `ttnn.transformer.exp_ring_joint_scaled_dot_product_attention` | Experimental ring-joint variant | tt_dit transformers |
| `ttnn.transformer.joint_scaled_dot_product_attention` | Joint streams, no ring | tt_dit transformers |
| `ttnn.experimental.nlp_create_qkv_heads` | **Fused QKV split + head reshape.** Replaces reshape+transpose chains — measured 1.45× on a ViT layer | LTX, Wan, Ideogram attention |
| `ttnn.experimental.nlp_concat_heads` / `ttnn.transformer.concatenate_heads` | The output-side counterpart | LTX, Wan |
| `ttnn.transformer.split_query_key_value_and_split_heads` | Fused packed-QKV split | tt_dit transformers |
| `ttnn.experimental.nlp_create_qkv_heads_vit` | ViT-shaped head creation | *bound, no tt_dit caller yet* — check it before hand-rolling ViT head ops |
| `ttnn.experimental.all_reduce_create_qkv_heads` | All-reduce fused with head creation | *bound, no tt_dit caller yet* |
| `ttnn.experimental.ring_attention_all_gather_async` | Ring-attention-shaped AG | *bound, no tt_dit caller yet* |

**Workflow rule.** `Permute`, `Reshape` or `Transpose` clustered around
attention is the head-op signature — 36% of one ViT layer's device time was
exactly this. Reach for `nlp_create_qkv_heads` / `nlp_concat_heads` before
writing anything.

---

## Rotary embedding

| Op | Does | Used by |
|---|---|---|
| `ttnn.experimental.rotary_embedding_llama` | Fused RoPE | Mochi, LTX, Ideogram attention and `rope_ltx.py` |
| `ttnn.experimental.rotary_embedding_hf` | HF-convention RoPE | tt_dit transformers |
| `ttnn.experimental.rotate_half` | Primitive, if you must compose | — |

---

## Convolution and spatial

| Op | Does | Used by |
|---|---|---|
| `ttnn.experimental.conv3d` + `prepare_conv3d_weights` | 3D convolution. `Conv3dConfig` carries the blocking surface — `utils/conv3d.py` is the tt_dit-side table | All video VAEs |
| `ttnn.conv1d` / `ttnn.conv2d` / `ttnn.conv_transpose2d` | 1D/2D convolution. All need `l1_small_size` | Audio VAEs, image VAEs |
| **`ttnn.experimental.neighbor_pad_async`** | **Halo padding across devices.** Padding values come from the neighbour device's shard, or from `padding_mode` (`zeros`, `replicate`) at the mesh edge. Supports **1D or fused 2D padding in one dispatch** (`dim=[2]` or `[2,3]`), per-dim `cluster_axis`, `num_links`, `persistent_output_buffer` | `parallel/manager.py`, `parallel/config.py` |
| `ttnn.upsample`, `ttnn.grid_sample` | Spatial resampling | VAE decoders |

**Workflow rule.** If you are writing halo exchange by hand for a sharded
convolution, use `neighbor_pad_async` — and use the **fused 2D form** when both
H and W are sharded rather than two 1D dispatches.

---

## CCL overlap with compute

The highest-leverage pattern in the tree, and the least obvious. A collective
next to a matmul is dead time; a collective *overlapped* with compute is free.
Three mechanisms compose, all wired through `parallel/manager.py::CCLManager`.
`models/transformers/ltx/attention_ltx.py` and `wan2_2/attention_wan.py` are the
reference implementations — read them before hand-rolling any of this.

**1. Reserve cores for the CCL so it physically runs concurrently.** Overlap is
not automatic: if the collective and the compute want the same cores they
serialize. LTX splits the grid, giving SDPA every column but the last and
placing the CCL workers on the reserved column:

```python
self.sdpa_worker_grid = (full_grid.x - 1, full_grid.y)   # compute grid, one column short
...
    program_config=...,                                   # uses sdpa_worker_grid
    ccl_core_grid_offset=(self.sdpa_worker_grid[0], 0),   # CCL lands on the spare column
    use_column_major_ccl=True,
```

If a profile shows a collective and a compute op that *should* overlap but their
durations still add up, this offset is usually missing.

**2. Ping-pong (double) buffers and semaphores** so iteration N+1's collective
can start while N is still in flight. `CCLManager` maintains an alternating pair
per (shape, dim, axis) and flips the index on every call:

| Getter | For |
|---|---|
| `get_ag_ping_pong_buffer` / `get_ag_ping_pong_semaphore` | All-gather |
| `get_rs_ping_pong_buffer` / `get_rs_ping_pong_semaphore` (+ `_fused`) | Reduce-scatter |
| `get_exp_ring_ping_pong_semaphore` | Experimental ring attention |
| `get_np_ping_pong_buffer` / `get_np_ping_pong_semaphore` | Neighbor pad (halo) |
| `get_sr_ping_pong_semaphore` | Slice-reshard |
| `get_fused_norm_stats_buffer`, `get_barrier_semaphore` | Distributed norms, barriers |

Convenience wrappers that take the persistent buffer for you:
`all_gather_persistent_buffer`, `reduce_scatter_persistent_buffer`,
`neighbor_pad_persistent_buffer`.

**3. Hand the collective a pre-allocated destination** — `persistent_output_buffer`
(or `_k` / `_v`). Without it the op must allocate before it can start, which
inserts a barrier exactly where you wanted overlap.

### Ops that fuse the collective into the compute

| Op | Fuses |
|---|---|
| `ttnn.transformer.ring_joint_scaled_dot_product_attention` | **The KV all-gather into attention itself.** Takes `persistent_output_buffer_k/v`, `multi_device_global_semaphore`, `cluster_axis`, `ccl_core_grid_offset`, `use_column_major_ccl`, `joint_strategy`, `logical_n` |
| Same op with `is_cross=True` | Cross-attention where Q is short and K/V is SP-sharded — the K/V gather folds into the ring SDPA and the output is the per-device Q shard |
| `ttnn.transformer.exp_ring_joint_scaled_dot_product_attention` | Experimental ring-joint variant (`get_exp_ring_ping_pong_semaphore`) |
| `all_gather_minimal_matmul_async` | AG into the column-parallel matmul |
| `minimal_matmul_strided_reduce_scatter_async` | RS into the row-parallel matmul |
| `dit_fused_distributed_{groupnorm,rmsnorm,layernorm}` | Stats all-gather into the norm (PRE → AG → POST) |

**Ring is not always the win.** LTX's masked audio self-attention deliberately
does *not* use ring-joint — an in-code comment records that gathering K/V and
running local SDPA beats it at that shape. Overlap has a cost; measure at the
real shape rather than assuming the fused path wins.

### Bare collectives

| Op | Notes |
|---|---|
| `ttnn.experimental.all_gather_async` | The workhorse — 25 call sites in tt_dit |
| `ttnn.experimental.reduce_scatter_minimal_async` | RS counterpart |
| `ttnn.experimental.slice_reshard_async` | Reshard fused with slice |
| `ttnn.experimental.send_async` / `recv_async` | Point-to-point |
| `ttnn.mesh_partition` | Partition a tensor across the mesh |
| `ttnn.experimental.all_gather_concat` | All-gather fused with head concatenation. Bound and tested (`tests/ttnn/unit_tests/operations/ccl/fusion_subtests/concat_fuse_test.py`); the source directory is named `all_gather_concat_heads_fused`, the Python name is not |

Prefer a fused collective+compute op over a bare collective adjacent to a
matmul. If you must use a bare one, still give it a persistent buffer.

---

## Ops with knobs you may not be using

The gap between what a config struct carries and what the call site passes is
where the cheapest wins live.

| Op | Knob worth checking |
|---|---|
| `minimal_matmul`, `all_gather_minimal_matmul_async` | `fused_activation=`, `fuse_swiglu=`, `bias_tensor=` — an unfused `Silu` or `Gelu` after a matmul means these are unset |
| `ttnn.matmul` | Fused activation via the program config |
| `Conv3dConfig` | `T_out_block`, `H_out_block`, `W_out_block`, `C_in_block`, `C_out_block`, `compute_with_storage_grid_size` |
| `init_device_compute_kernel_config` | `math_fidelity`, `fp32_dest_acc_en`, `packer_l1_acc`, `math_approx_mode` — fidelity is the 2× lever, `packer_l1_acc` frees accumulation pressure |
| `dit_fused_distributed_rmsnorm` / `_layernorm` | `persistent_output_buffer`, `topology`, plus the fold-ins above |
| `neighbor_pad_async` | fused 2D (`dim=[2,3]`), `num_links`, `persistent_output_buffer` |
| SDPA | chunk size — a ViT decoder measured 2.95× at `q=k=192` over defaults |

---

## Classifying a hot spot

| Profile shows | Likely fast path |
|---|---|
| `Untilize` + `Tilize` around `GroupNorm` | Layout round-trip. No fused distributed GroupNorm exists — compute stats in the neighbours' layout instead of round-tripping |
| `AllGather` between two norm halves | `dit_fused_distributed_{groupnorm,rmsnorm,layernorm}` |
| `RMSNorm` → `Silu`/`Gelu` | `dit_rms_norm_unary_fused` |
| `Matmul` → `BinaryNg` chain in an AdaLN block | `dit_minimal_matmul_addcmul_fused` |
| `AllGather`/`ReduceScatter` adjacent to `Matmul` | `all_gather_minimal_matmul_async`, `minimal_matmul_strided_reduce_scatter_async` |
| `Permute`/`Reshape`/`Transpose` around attention | `nlp_create_qkv_heads`, `nlp_concat_heads` |
| Hand-rolled halo slicing before a sharded conv | `neighbor_pad_async` (fused 2D) |
| `Matmul` → `Silu`/`Gelu` | `fused_activation=` on the matmul |
| A collective and a compute op whose durations **add up** instead of overlapping | Missing `ccl_core_grid_offset` — they're fighting for the same cores |
| `AllGather` before SDPA in a sequence-parallel block | `ring_joint_scaled_dot_product_attention` (add `is_cross=True` for short-Q cross-attention) |
| A collective preceded by an allocation stall | Missing `persistent_output_buffer` / ping-pong buffer |
| Elementwise at HiFi4 | Fidelity, not fusion |

If nothing matches, then it is genuine kernel work — hand off to
`../tt-dit-kernel-research/SKILL.md` with the profile attached, and record in the
journal what you searched and did not find.
