# Parallelism

The mesh is the resource. Reach for parallelism **before** kernel tuning — a 30×
from spreading work across a Galaxy dwarfs a 1.7× from a blocking sweep, and the
two compose.

## Config types

`parallel/config.py`. Each is a `NamedTuple` of `ParallelFactor(factor, mesh_axis)`.

| Config | Axes | Used by |
|---|---|---|
| `DiTParallelConfig` | `cfg_parallel`, `tensor_parallel`, `sequence_parallel` | DiT transformers |
| `EncoderParallelConfig` | `tensor_parallel`, optional `sequence_parallel` | Text encoders |
| `VAEParallelConfig` | `tensor_parallel` | Simple VAEs |
| `VaeHWParallelConfig` | `height_parallel`, `width_parallel` | Spatially sharded VAEs |
| `MochiVAEParallelConfig` | `time_parallel`, `h_parallel`, `w_parallel` | 3-axis video VAE |
| `AudioTParallelConfig`, `AudioTCParallelConfig` | time, channel | Audio VAEs / vocoders |

A config is meaningless without its mesh shape. On `2×4` with `sp_axis=0,
tp_axis=1` → SP=2, TP=4; on `4×8` with `sp_axis=1, tp_axis=0` → SP=8, TP=4. Test
ids encode it: `bh_2x4sp1tp0`, `bh_4x8sp1tp0_ring`.

Design every module so `factor == 1` is the serial path with no collectives —
`vae_all_gather` short-circuits when the cluster axis has extent 1, which is what
lets one code path run on a single device.

## Choosing

| Order | Kind | Buys | Cost |
|---|---|---|---|
| 1 | **Data parallel over work units** | ~30× near-linear on a 4×8 | None — no collectives, no halo. **Bit-exact** |
| 2 | **Spatial H/W sharding** | Latency on a *single* unit, which DP cannot buy | Halo exchange on every conv with kernel > 1 |
| 3 | **Sequence parallel** (ring attention) | Long-sequence DiT scaling | KV all-gather — overlapped with compute, not free |
| 4 | **Tensor parallel** | Wide hidden dims, large weights | AllGather / ReduceScatter every layer |
| 5 | **CFG parallel** | Models that run CFG | Halves the group available to SP |

Check DP first — it gets skipped because it feels too easy. But it only applies
when the input is *already* chopped into independent units; when one unit owns
the whole mesh, (2) onward is the only route.

## Spatial H/W sharding — the LTX VAE pattern

`models/vae/vae_ltx.py` is the reference; `vae_wan2_1.py` is the pattern it
mirrors. Height across one mesh axis, width across the other, via
`VaeHWParallelConfig`.

| Piece | Detail |
|---|---|
| **Padding split** | External `neighbor_pad` when sharded, internal conv padding when not. `height_parallel.factor > 1` / `width_parallel.factor > 1` select between them per axis |
| **Halo exchange** | `ttnn.experimental.neighbor_pad_async`, or `CCLManager.neighbor_pad` / `neighbor_pad_persistent_buffer`. Fused 2D form (`dim=[2,3]`) does H and W in **one dispatch** — use it when both axes are sharded |
| **Semaphores** | `get_np_ping_pong_semaphore(mesh_axis)` — double-buffered so the next halo can start while the current one is in flight |
| **Conv threading** | `h_factor` / `w_factor` pass into `ttnn.experimental.conv3d`; `utils/conv3d.py` has `conv_pad_height`/`conv_unpad_height`/`conv_pad_width`/`conv_unpad_width` and `compute_encoder_dims`/`compute_decoder_dims` for the shape arithmetic |
| **Width padding mask** | **`neighbor_pad` has no W mask.** LTX keeps a cached mask that zeros width-padding columns beyond `logical_w` *before* the halo. Skip it and garbage from the padded region propagates into the convolution |
| **Logical vs padded extent** | Track `logical_w` separately from the sharded/padded width — they diverge as soon as the width isn't divisible by the factor |

**Halo bugs are quiet.** They produce small error concentrated at shard seams
that a whole-tensor PCC waves through. Gate the sharded path against the
unsharded path at the production shape and slice the seam rows separately.

## Sequence × tensor parallel

The two DiT axes compose, and both appear in `models/transformers/wan2_2/` and
`ltx/`.

| | Fractures | Collective | Where it lands |
|---|---|---|---|
| **SP** | The token sequence, across `sequence_parallel.mesh_axis` | KV all-gather, **overlapped with attention compute** via ring attention | `ring_joint_scaled_dot_product_attention` |
| **TP** | Weights, across `tensor_parallel.mesh_axis` | AllGather / ReduceScatter on activations every layer | Column-parallel and row-parallel linears |

Column-parallel produces a sharded output that the next op consumes directly;
row-parallel reduce-scatters. Prefer the **fused** forms —
`all_gather_minimal_matmul_async` and `minimal_matmul_strided_reduce_scatter_async`
— over a bare collective next to a matmul.

RoPE tables must be sliced to the SP shard
(`rope_cos_1HND, num_devices=sequence_parallel.factor`), and FSDP, when enabled,
shards along the **SP** mesh axis (`fsdp_mesh_axis = sequence_parallel.mesh_axis`).

Text encoders are usually TP-only. The widest mesh axis should carry whichever
of SP/TP the model is more sensitive to — benchmark rather than assume.

## Collectives

`parallel/manager.py::CCLManager` owns semaphores, ping-pong buffers and
plumbing. Ring topology (`ttnn.FabricConfig.FABRIC_1D_RING`) is available on 4×8
and is what LTX uses.

If AllGather or ReduceScatter shows up in the top ops, the first question is not
"which axis" but **"is it overlapping with compute?"** — see
`../tt-dit-benchmark-profile/existing-fast-paths.md` § "CCL overlap with compute"
for core reservation, ping-pong buffers and the fused collective+compute ops.

## GroupNorm

Sharded GroupNorm is the most common source of hangs in this tree. Read
`known-issues.md` before wiring one.

Note there is **no fused distributed GroupNorm** — RMSNorm and LayerNorm have
one (`dit_fused_distributed_{rmsnorm,layernorm}`), GroupNorm does not.
`layers/normalization.py::GroupNorm3D` drives the local `ttnn.group_norm` with a
grid pinned by `determine_expected_group_norm_dram_grid_size`; cross-device
statistics are yours to assemble.
