# Neighborhood attention in the LTX-2.5 DiffVAE decoder

A map of every file involved in 3D neighborhood attention (NA) and its strided
generalization (GNA), across **two independent implementations** that live side by side:

- **the reference implementation** — `na3d.py` plus parameters on the shared SDPA op. Older,
broader, feeds every stage of the decoder.
- **ours** — a self-contained `neighborhood_sdpa` op with all geometry in one host-testable
file. Newer, stage-5 only, selected by backend name.

Read the terminology first: the two implementations use *different words for the same things*,
and that is the single biggest obstacle to reading them together.

---



## 1. Terminology



### The technique


| Term                            | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Neighborhood attention (NA)** | Each query attends only to a fixed-size box of keys around itself, instead of to everything. Cost goes from `O(N²)` to `O(N·k³)`.                                                                                                                                                                                                                                                                                                |
| **context window**              | The box of keys one query group attends to. `11×11×11` at stage 5. The paper's term; ours uses it everywhere. The reference calls it `kernel` / `kernel_size` / `k`.                                                                                                                                                                                                                                                             |
| **the clamping rule**           | The rule that makes NA tractable: at a volume boundary the window keeps its **size** and slides **inward**, it does not truncate. A query at site 0 attends to `[0, K)`, not to a half-empty `[0, K/2]`. Consequences: every query attends to the same number of keys, every query is inside its own window, and there is never anything out of range to mask. A truncating window looks plausible and is wrong near every edge. |
| **GNA (generalized NA)**        | Queries are grouped; a whole group shares one window instead of each query being centred on its own.                                                                                                                                                                                                                                                                                                                             |
| **stride**                      | The extent of a query group, per axis. `(1,1,1)` is plain NA — every query centred on itself. Larger means fewer distinct windows and less work, but queries at the edge of a group get an off-centre receptive field, **which changes what the model computes**.                                                                                                                                                                |
| `b == s`                        | Block equals stride: the regime where a group's gathered box collapses to exactly the kernel. Both implementations chase this; it is what makes the mask broadcast and the gather stop growing.                                                                                                                                                                                                                                  |




### Layout


| Term                  | Whose     | Meaning                                                                                                                                                                       |
| --------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Volume**            | ours      | The `(time, height, width)` token grid, in sites.                                                                                                                             |
| **Site**              | ours      | One `(time, height, width)` position.                                                                                                                                         |
| **Brick**             | ours      | A compact 3D box of exactly **32 sites** = one hardware tile row. Named "brick" because *block* already means transformer block, matmul block, and block-sparse in this tree. |
| **Natural / Bricked** | ours      | Token order. Natural is row-major `(t,h,w)`; bricked puts 32 spatially-adjacent sites in one tile row.                                                                        |
| **query chunk**       | ours      | The set of queries sharing **one gather**. Derived from the stride, never tuned: a multi-brick chunk must equal the stride exactly.                                           |
| **block**             | reference | Their equivalent of chunk+brick combined — a `(bt,bh,bw)` box of queries reordered to be contiguous. See `_pick_block`.                                                       |
| **op order**          | reference | Their kernel works in `(W, H, T)` axis order, **not** `(t,h,w)`. Physical strides are permuted on the way in. This is why an error about `t=8` can really be about *width*.   |
| **band / slab**       | reference | A range of frames processed together to bound peak memory. `DIFFVAE_SLAB_FRAMES`.                                                                                             |
| **halo**              | both      | The border sites a shard or band needs from its neighbour because windows reach across the boundary.                                                                          |
| **shard**             | both      | One device's portion of the volume. Here always split along **W**.                                                                                                            |




### Cost model


| Term            | Meaning                                                                                                                                                                                                     |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **gather**      | The keys one query chunk must read: the union of its queries' context windows, rounded out to whole bricks.                                                                                                 |
| **box**         | Reference's name for the same thing.                                                                                                                                                                        |
| `box / vol`     | Keys gathered per query. Governs **gather traffic**. Ours logs it as `keys/query`.                                                                                                                          |
| **box** (alone) | Governs **score compute** — every query scores against the whole box.                                                                                                                                       |
| **regime**      | Where a chunk sits against a volume boundary on one axis: `Low` (clamps to 0), `Interior` (centred), `High` (clamps to the far edge). 3 per axis → **27 distinct mask geometries** in a volume of any size. |
| **coverage**    | How much of a gathered key brick a query chunk can see: `AllVisible`, `NoneVisible`, or `Mixed`. Uniform bricks are constant-filled; only `Mixed` needs per-site evaluation.                                |


---



## 2. Our implementation



### 2.1 Geometry — the load-bearing layer

Pure C++. No ttnn, no kernel, no device includes, on purpose: the geometry is where the bugs
are, and this way it is testable on the host against a brute-force oracle with no hardware.

#### `ttnn/.../sdpa/device/neighborhood_plan.hpp` (~200 lines)

The vocabulary. Everything else imports its nouns from here.

```cpp
constexpr uint32_t SITES_PER_BRICK = 32;
enum class Axis   { Time, Height, Width };
enum class Order  { Natural, Bricked };
enum class Regime { Low, Interior, High };

struct Extent3 { std::array<uint32_t,3> by_axis; };   // a SIZE, in sites
struct Site    { std::array<uint32_t,3> by_axis; };   // a POSITION, in sites
struct Offset3 { std::array<int32_t,3>  by_axis; };   // a SIGNED position -- see sharding
```

`Extent3` and `Site` are deliberately distinct types so a size cannot be passed where a
position belongs.

`NeighborhoodConfig` — the complete description of one problem:


| field                | meaning                                                            |
| -------------------- | ------------------------------------------------------------------ |
| `volume`             | the **global** token grid                                          |
| `context_window`     | what one query group attends to                                    |
| `stride`             | query group extent                                                 |
| `brick`              | layout unit; `brick.sites() == 32`                                 |
| `query_chunk_bricks` | how many bricks share one gather                                   |
| `shard_extent`       | what this device holds (owned + halo); zero means "same as volume" |
| `shard_origin`       | **signed** — where that sits in the global volume                  |


`NeighborhoodPlan` — what `build_plan` produces, cached because it uploads index tables:


| field                                           | meaning                                                                    |
| ----------------------------------------------- | -------------------------------------------------------------------------- |
| `volume_bricks`, `brick_count`                  | the volume measured in bricks                                              |
| `volume_chunks`, `chunk_count`                  | the volume measured in **work items**                                      |
| `gather_extent`, `gather_sites`, `gather_tiles` | site-exact gather                                                          |
| `gather_bricks`, `gather_brick_count`           | rounded out to whole bricks — what a tile-granular read can actually fetch |
| `gather_origin_by_chunk`                        | one origin per chunk, rounded **down** to a brick boundary                 |




#### `neighborhood_plan.cpp` (~350 lines)


| function                                          | does                                                                                                                                                                                                          |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `choose_brick(window)`                            | Enumerates the eight factorisations of 32, returns the one minimising `∏(window_axis + brick_axis − 1)`. **A function, not a constant**: an 11³ window wants `(2,4,4)`; a `(1,11,11)` window wants `(1,4,8)`. |
| `context_window_for(site, config)`                | Where one group's window starts.                                                                                                                                                                              |
| `validate_config(config)`                         | Throws on a config that cannot be built. **Named** `validate_config` **not** `validate` because a repo hook pattern-matches a member called `validate` as a legacy device op.                                 |
| `build_plan(config)`                              | Everything above, assembled.                                                                                                                                                                                  |


Key internals and their loop variables:

```cpp
for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index)
for (uint32_t chunk_index = 0; chunk_index < plan.chunk_count; ++chunk_index)
```

- `gather_extent_on_axis(chunk_extent, stride, window, volume)`
`= window + (ceil(chunk/stride) − 1) · stride`, clamped to the axis.
When `chunk == stride` this is **exactly the window**, however large the chunk grows. That is
the whole reason a big chunk is cheap rather than merely big.
- `gather_bricks` is **measured** — the max misalignment across all chunks — not bounded
conservatively.
- Gather origins are rounded down to brick boundaries and translated to **local** coordinates by
subtracting `shard_origin`.

**The invariant that makes multi-brick chunks legal** (in `validate_config`):

```
if (bricks_per_query_chunk() > 1)
    require(query_chunk_sites() == stride)
```

Violating it is *silently wrong*, not loud: the kernel applies the first row's mask to every
row, so queries attend to a window that is not theirs and still return plausible video.

#### `ttnn/.../sdpa/device/kernels/neighborhood_window_rule.hpp` (~120 lines)

**The single definition of the window rule**, included by the host planner, the device mask
generator, *and* transcribed into Python. Two transcriptions that drift produce a kernel that
reads the wrong keys and still returns plausible video — hence one file, with no includes
beyond `<cstdint>`.


| function                                                              | does                                                                                                                               |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `window_origin_on_axis(group_index, stride, window, volume, brick=0)` | Where the window starts. Centres on the group, then slides inward rather than truncating. `brick != 0` enables **brick snapping**. |
| `key_is_in_window_on_axis(...)`                                       | Whether one key is inside one query's window.                                                                                      |
| `snap_extent_on_axis(stride, brick)`                                  | Whether snapping is legal on this axis, and to what.                                                                               |


**Brick snapping** deserves its own note. Centring is not the only legal placement: any origin
that keeps the window in bounds and still contains the whole group is valid NA. An unaligned
origin makes a window straddle one extra brick per axis — a 12-site window from an origin
`3 mod 4` spans 4 bricks instead of 3. Snapping picks an aligned one when the legal range
allows. Worth **54 gathered bricks instead of 96** at an 11³ window.

`snap_extent_on_axis` is legal exactly when a whole brick lies inside one query group, i.e.
`stride % brick == 0`. This rule was open-coded in **five** places before being centralised;
that is the drift hazard this header exists to prevent.

### 2.2 The host↔device argument layout



#### `kernels/neighborhood_kernel_args.hpp` (~240 lines)

CB ids and compile-arg slots **named on both sides**, so the factory and the kernel cannot
disagree about what argument 14 means.

```cpp
enum CircularBufferId {
  cb_query, cb_key, cb_value, cb_mask, cb_reduce_scalar, cb_zero, cb_column_identity,
  cb_scores, cb_row_max_current, cb_row_max_previous, cb_row_sum_current, cb_row_sum_previous,
  cb_exp_max_difference, cb_output_accumulator_current, cb_output_accumulator_previous,
  cb_output, cb_gather_origin, cb_resident_mask, CB_COUNT };
```

`gather_origin_column` — the table's layout. The row is padded to 64 B so each chunk's entry
is one DRAM-aligned page, which leaves spare columns:


| column | carries                                                     |
| ------ | ----------------------------------------------------------- |
| 0–2    | this chunk's gather origin, in **local** sites              |
| 3–5    | this device's shard origin, in **global** sites, **signed** |


The shard origin rides here because it is the one geometric value that **differs per device**,
and a mesh runs one program. As a compile-time argument it would be uniform across the mesh, so
every shard would believe it sat at the origin and clamp its windows at its own seam.

Also: `reader_arg`, `compute_arg`, `writer_arg` enums, and

```cpp
struct AxisExtents        { uint32_t time, height, width; };
struct SignedAxisOffsets  { int32_t  time, height, width; };
struct NeighborhoodExtents { AxisExtents brick_sites, context_window, stride, volume,
                             query_chunk, resident; SignedAxisOffsets shard_origin; };
```



#### `kernels/neighborhood_chunk_layout.hpp` (~90 lines)

Small named helpers shared by reader and writer, so the two cannot decode a chunk differently:


| helper                                                                     | does                                                |
| -------------------------------------------------------------------------- | --------------------------------------------------- |
| `linear_to_point3(index, grid)`                                            | linear index → `BrickCoordinate`                    |
| `chunk_origin_brick(chunk_index, volume_chunks, chunk_bricks)`             | which brick a chunk starts at                       |
| `brick_within_chunk(index_in_chunk, origin, chunk_bricks)`                 | the n-th brick of a chunk                           |
| `brick_index(brick, volume_bricks)`                                        | `BrickCoordinate` → linear                          |
| `brick_is_inside(brick, volume_bricks)`                                    | a chunk on the far edge can hang off the volume     |
| `tile_offset(batch, brick, head, brick_count, head_count, head_dim_tiles)` | **the one place that knows tensors are site-major** |


**Site-major** means `[batch, 1, bricked_sites, head_count * head_dim]`: sites are the tile
**row** axis, heads are columns. Head-major would put a head's sites contiguously, but nothing
here reads a head contiguously — every read is one brick of one head — and it would force the
caller to transpose heads against sites on the way in and back on the way out. That transpose
measured **24.6 ms per block** at stage-5 size for no arithmetic.

Note ttnn tiles the **last two** dimensions, which is why the shape is `[b, 1, sites, channels]`
and not `[b, sites, heads, head_dim]` — the latter would cut tiles across heads.

### 2.3 The op — ttnn plumbing, no geometry


| file                                           | lines | contains                                                                                                                                                                                                                    |
| ---------------------------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `neighborhood_sdpa_device_operation_types.hpp` | ~70   | `NeighborhoodSDPAParams` (config, `head_count`, scale, `tiles_per_kv_chunk`, kernel config, output memory config) and `NeighborhoodSDPAInputs` (q, k, v, `gather_origin_table`, optional `interior_mask`, optional output). |
| `neighborhood_sdpa_device_operation.{hpp,cpp}` | ~260  | `validate` and `compute_output_specs`. Explicit `compute_program_hash` that **excludes** `shard_origin`, so one compiled program serves every shard.                                                                        |
| `neighborhood_sdpa_program_factory.cpp`        | ~290  | CB allocation, core assignment, compile args.                                                                                                                                                                               |
| `neighborhood_sdpa_nanobind.cpp`               | ~200  | Python bindings: `neighborhood_choose_brick`, `neighborhood_plan`, `neighborhood_scaled_dot_product_attention`.                                                                                                             |


`head_count` is an explicit parameter because it **cannot be read off the shape** — the tensor
is `[b, 1, sites, heads*head_dim]` and heads are folded into the column axis.

Program factory, key locals:

```cpp
head_count      = attributes.head_count;
head_dim_tiles  = query_shape[3] / head_count / TILE_WIDTH;
query_tile_rows = config.bricks_per_query_chunk();       // M, in tile rows
tiles_per_kv_chunk                                        // N per flash step, bounded by DST
kv_chunk_count  = ceil_div(plan.gather_brick_count, tiles_per_kv_chunk);
work_item_count = batch_count * head_count * plan.chunk_count;
```

**DST capacity** is the constraint that shapes everything: 8 tiles (4 with fp32 accumulate). A
chunk wider than DST silently returns wrong numbers rather than faulting, so subblock widths are
derived from it and `validate` enforces the bound.

### 2.4 Kernels



#### `kernels/dataflow/neighborhood_reader.cpp` (~370 lines)

**The only kernel that knows what a context window is.** Feeds one query CHUNK at a time.

Work item decomposition (must match the writer exactly):

```cpp
chunk_index = work_item % chunk_count;
head_index  = (work_item / chunk_count) % head_count;
batch_index =  work_item / (chunk_count * head_count);
```

Loops:

```cpp
for (work_item ...)                                     // one query chunk, one head
  for (brick_in_chunk < bricks_per_query_chunk)         // Q: one tile row per brick
  for (kv_chunk_index < kv_chunk_count)                 // flash steps
    for (slot < tiles_per_kv_chunk)                     // classify + read K/V
    for (slot < tiles_per_kv_chunk)                     // uniform bricks: constant fill
    for (slot < tiles_per_kv_chunk)                     // Mixed bricks: fetch or generate
```

**Three separate passes over** `slot`**, deliberately.** Mixing the constant fills with
`fill_mask_tile` in one loop body puts a large function (nested loops, divisions) next to a
memset in the same instruction cache and measured **worse than either alone** — 7498 ms against
2761 ms for generating everywhere.

**K and V are laid out differently in their CBs, and it matters.** `matmul_blocks` walks `in1` as
`in1_index += N`, so `in1` is always a `[K, N]` grid of tiles — the `transpose` flag transposes
each **tile**, not the grid. So:

- **K** must be head-dim-major: `[head_dim_tiles][slots]`
- **V** is `[slots][head_dim_tiles]`, which is what the gather naturally produces

At `head_dim_tiles == 1` the two layouts are the same buffer. That is why a wrong K layout
survived every test until one used a 64-wide head — and stage 5 *is* 64-wide.

Other locals: `gather_origin_brick` (decoded from the table), `chunk_origin_site`,
`coverage[]`, `key_origins[]`, `resident_regime` (which regime's set is currently in
`cb_resident_mask`).

#### `kernels/dataflow/neighborhood_mask_gen.hpp` (~250 lines)


| function                              | does                                                                    |
| ------------------------------------- | ----------------------------------------------------------------------- |
| `to_global_site(local, shard_origin)` | the one named local→global conversion; clamps below-volume halo columns |
| `key_is_visible(...)`                 | one key vs one query, via the shared window rule                        |
| `classify_brick(...)`                 | → `AllVisible` / `NoneVisible` / `Mixed`                                |
| `fill_mask_tile(...)`                 | the per-site path, for `Mixed` only                                     |


`fill_mask_tile` is force-inlined and hoists window resolution out of the column loop; key
column positions are precomputed once per tile. That hoist alone measured 1.35–1.83×.

**Ghost handling:** ghost *keys* are always masked. Ghost *query rows* are left open — an
all-`-inf` row softmaxes to NaN, which propagates through the flash rescale into real rows.

#### `kernels/compute/neighborhood_sdpa.cpp` (~170 lines)

Standard flash attention. **Contains no neighborhood concepts at all** — this was an explicit
design invariant, and it held. If `context_window`, `stride`, `brick`, `Volume` or `Regime` ever
appears here, the design has leaked.

```
QK^T (+mask) → running row max → exp → accumulate PV → running row sum → normalise
```

Ping-pong buffers rather than copies: `current_max`/`previous_max`, `current_sum`/`previous_sum`,
`current_output`/`previous_output`, swapped each chunk.

Two subtleties that cost real debugging:

- `matmul_blocks` pops `in1` but leaves `in0` produced, so `cb_scores` must be retired
explicitly or the next chunk deadlocks.
- The running max is the one statistic nothing else retires. Leaking one tile per work item
jams on the **third** item — invisible while every test had exactly one item per core.



#### `kernels/dataflow/neighborhood_writer.cpp` (~120 lines)

Writes output tiles in bricked order using the same chunk decomposition as the reader, skipping
bricks that hang off the volume. Also generates the reduce scalar, the zero tile, and the column
identity used by `matmul_reduce`.

### 2.5 Python



#### `models/tt_dit/layers/neighborhood_permute.py` (~175 lines)

`to_bricked` / `to_natural` as reshape + permute in **ROW_MAJOR**. Measured: ROW_MAJOR permute
0.48 ms per 50 MB against 7.35 ms in TILE — 15×. Padding uses `ttnn.concat` of zeros because
`ttnn.pad` cannot pad dim 1 of a rank-5 tensor.

#### `models/tt_dit/layers/neighborhood_reference.py` (~200 lines)

The torch definition of correct. `context_window_origin`, `snap_extent`, `neighborhood_mask`,
`neighborhood_attention_3d`. Transcribed from the C++ rule and checked against the same
search-based oracle, which is what keeps them from drifting.

#### `models/tt_dit/layers/neighborhood_attention.py` (~500 lines)

The executor. Two entry points:


| function                                      | for                        |
| --------------------------------------------- | -------------------------- |
| `neighborhood_attention_3d_bricked`           | single device / replicated |
| `neighborhood_attention_3d_bricked_w_sharded` | W-sharded across the mesh  |


Helpers worth knowing:


| helper                                  | does                                                                                                                                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `configured_stride()`                   | `DIFFVAE_S5_GNA_STRIDE` then `DIFFVAE_GNA_STRIDE`. The stage-5 knob exists because the global one is read by `na3d` for **every** stage, and the deterministic stages have smaller kernels. |
| `_query_chunk_bricks(stride, brick)`    | `stride // brick` per axis where it divides, else 1. **Derived, never tuned.**                                                                                                              |
| `halo_sites(window, brick)`             | `ceil(window/2 / brick) * brick` — the halo, in whole bricks                                                                                                                                |
| `_tiles_per_kv_chunk(gather)`           | largest chunk that fits DST and divides the gather                                                                                                                                          |
| `_build_regime_masks(...)`              | the 27 uploaded mask sets, enumerated per **chunk**                                                                                                                                         |
| `_cached_plan(...)`                     | plan + uploaded tables, cached per geometry; unsharded is the one-shard case                                                                                                                |


The sharded path builds **one plan per shard** and stacks the tables, uploading with
`mesh_axes=[sp_axis]` so each device reads its own origins. It asserts the plans agree on
`chunk_count`, `gather_brick_count`, `gather_bricks`, `volume_chunks` — they must, because one
program serves the mesh.

### 2.6 Tests


| file                                                      | covers                           |
| --------------------------------------------------------- | -------------------------------- |
| `tests/ttnn/unit_tests/gtests/test_neighborhood_plan.cpp` | geometry vs a brute-force oracle |
| `models/tt_dit/tests/unit/test_neighborhood_permute.py`   | bricked↔natural round-trip       |
| `models/tt_dit/tests/unit/test_neighborhood_reference.py` | the torch reference itself       |
| `models/tt_dit/tests/unit/test_neighborhood_sdpa.py`      | the op vs torch — 26 cases       |
| `models/tt_dit/tests/unit/test_neighborhood_sdpa_perf.py` | scale timing, no correctness     |


The op test's parametrisation is where the coverage lives:

- `stride_one`, `stride_equals_brick`, `partial_gather`, `many_items_per_core`,
`multi_brick_chunk`, `chunk_overhangs_volume`
- `widest_chunk` / `narrow_chunk` — whether the online rescale runs
- `one_tile_row` **/** `two_tile_row` — `head_dim` 32 vs 64. Added after the K-layout bug.
- `test_shards_match_the_whole_volume` — two shards, different origins, same program
- `test_symmetric_halo_shards_match_the_whole_volume` — three shards including a **negative** origin

---



## 3. The reference implementation



### `models/tt_dit/layers/na3d.py` (~1530 lines)

The hub. Contains the torch reference, the planner, the sharding descriptors, and **six**
device executors selected by string.


| symbol                                                          | is                                                            |
| --------------------------------------------------------------- | ------------------------------------------------------------- |
| `window_bounds(length, kernel, stride)`                         | their clamping rule — returns per-site window `starts`/`ends` |
| `plan_na3d`, `NA3DPlan`, `TileGroup`                            | group queries into tiles, build masks                         |
| `na3d_torch`                                                    | torch reference                                               |
| `NA3DShard`, `NA3DGroup`, `NA3DDevicePlan`, `build_device_plan` | multi-device query sharding                                   |
| `_pick_block(t_full, h_full, w_local, kmax, gna)`               | **the block/stride chooser**                                  |
| `neighborhood_attention_3d(...)`                                | the dispatcher; `backend=` selects one of the below           |


Executors, in rough order of specialisation:


| backend                                | function                                                                          |
| -------------------------------------- | --------------------------------------------------------------------------------- |
| `"gather"`                             | the grouped gather + dense masked attention (default)                             |
| `"op"`                                 | `neighborhood_attention_3d_op` — mask synthesised inside SDPA, no gather          |
| `"fused"`                              | `neighborhood_attention_3d_op_fused`                                              |
| `"op_sp"`                              | `..._op_sp` — attention split over T                                              |
| `"op_sp_w_sharded"`                    | `..._op_sp_w_sharded` — **the fast path**; W-sharded, fused kernel, block permute |
| `"bricked"` / `"bricked_sp_w_sharded"` | **ours**                                                                          |


`_pick_block` **is worth reading in full.** It searches `(bt,bh,bw)` where each divides its axis,
`vol` is a multiple of 32 in `[128, 512]`, and with `gna=True` each dim is capped at `kmax`. The
objective inverts between modes:

```python
key = (-vol, box) if gna else (box, -vol)
```

With GNA the box is the kernel on every axis regardless of block shape, so volume — which sets
the chunk count — becomes the whole objective. Its docstring carries a **measured quality
warning**:

> *"This objective is purely a speed objective and it is not free… block (11,4,8) vs stride-1
> attention: PCC 0.51 on iid Q/K/V, 0.72 at a spatial correlation length of 8 tokens… A network
> trained at stride 1 cannot absorb the picked block — constrain the stride, or leave GNA off,
> unless retraining."*

**Axis order.** `op_stride = (sw, sh, st) if t_inner else (sw, st, sh)`. The kernel works in
`(W,H,T)` order. This is why a `TT_FATAL` about `stride t=8` can actually be about *width*.

### `models/tt_dit/models/vae/diffvae_ltx_stage5.py` (~1600 lines)

Stage 5: 8 diffusion NA blocks on the largest grid, where almost all decode compute lives.


| symbol                                              | is                                                                |
| --------------------------------------------------- | ----------------------------------------------------------------- |
| `Grid`                                              | `(batch, t, h, w)` — always the **full** grid, even when sharded  |
| `DiffVAEStage5Config`                               | includes `kernel_size` and `gna_stride`                           |
| `_Band`, `_bands(t, frames, kernel)`                | frame banding for peak memory; halo from `window_bounds`          |
| `_RopeTables`, `_build_rope_tables`, `_apply_rope`  | RoPE, stored **factored**                                         |
| `_NeighborhoodAttention3D.forward(y, grid, tables)` | the per-block attention; one `match` on `self.kernel.name`        |
| `DiffusionNABlock`                                  | context inject → AdaLN residual attention → AdaLN residual SwiGLU |
| `DiffVAEStage5.forward_diff_step(...)`              | the 8-block loop, per band                                        |
| `NAKernel`, `_NA_KERNELS`, `resolve_na_kernel()`    | the backend record: W-shard, bricking, flat sequence              |


**The RoPE factorisation is a real constraint on any future work.** A row of the fused table is
`[T-lanes | H-lanes | W-lanes]`, so the H/W lanes repeat every frame and the T lanes repeat at
every site within a frame. Storing the two pieces rather than their combination keeps the table
off the critical memory path: the volume form is **9.7 GB per table** at 6 s 1920×1088.
`_apply_rope` evaluates `x*cos` as `x*frame.cos + x*time.cos`.

Consequence: **a brick two deep in time mixes two frames into one tile row and destroys that
factorisation.** Any hoist of the brick permute out of the per-block loop needs a *time-flat*
brick like `(1,4,8)`.

`_wshard_context` reshards the replicated context to W-sharded. Its own docstring notes the
replicated form is 9.7 GiB at 6 s — that allocation is what OOMs a 145-frame 1080p decode when
banding is off.

### `models/tt_dit/models/vae/diffvae_ltx.py` (~1340 lines)

The decoder around stage 5.


| symbol                       | is                                                               |
| ---------------------------- | ---------------------------------------------------------------- |
| `decoder_config(path)`       | reads shapes from the checkpoint                                 |
| `rope_tables`, `apply_rope`  | the deterministic stages' RoPE                                   |
| `NeighborhoodAttention`      | the det-stage attention block                                    |
| `NABlock`, `SwiGLU`          | det-stage transformer block                                      |
| `LinearPixelShuffleUpsample` | between stages; slabs its projection to bound the widened copy   |
| `DeterministicStages`        | stages 1–4 plus `conv_in` (latent denormalisation folded in)     |
| `DiffVAEDecoder`             | the whole decoder; `stages_na3d_backend` / `stage5_na3d_backend` |




### `models/tt_dit/layers/block_permute.py` (~150 lines)

Their equivalent of our bricking: reorder tokens so a block is contiguous.
`to_block_order` / `from_block_order` (torch) and `_tt` variants (device).

### `models/tt_dit/utils/decode_tree.py` (~275 lines)

The timing tree. `open_span` / `close_span`, `Node`, `roots()`, `render()`. Spans nest by a
thread-local stack; siblings with the same label are pooled with an `n` count.
`ENABLED` ← `DIFFVAE_STAGE_TIMING`, `DEEP` ← `DIFFVAE_BLOCK_PROF`.
**Not valid under trace capture**, and absolute totals are inflated by one
`synchronize_device` per span open/close.

### Their SDPA kernels


| file                                       | lines | does                                                                      |
| ------------------------------------------ | ----- | ------------------------------------------------------------------------- |
| `kernels/windowed_loop_geometry.hpp`       | 395   | per-chunk window bounds, computed as templates in both reader and compute |
| `kernels/dataflow/windowed_mask_gen.hpp`   | 861   | their mask generator                                                      |
| `kernels/dataflow/neighborhood_gather.hpp` | 192   | the fused reader's gather                                                 |
| `kernels/dataflow/sparse_sdpa_gather.hpp`  | 103   | MLA-oriented gather, borrowed patterns                                    |
| `device/sdpa_program_factory.cpp`          | 1603  | the shared SDPA factory, with NA parameters threaded through              |


`sdpa_device_operation.cpp:549` is where `stride[i] <= k_eff` lives — the check that rejects a
stride larger than its kernel, reported in **op-order axes**.

---



## 4. Environment variables



### Ours


| variable                    | does                                                                                          |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| `DIFFVAE_S5_GNA_STRIDE`     | stage-5 stride, physical `(t,h,w)`. **Prefer this** — the global one leaks into other stages. |
| `DIFFVAE_GNA_STRIDE`        | global stride; read by `na3d` for every stage                                                 |
| `DIFFVAE_NA_WINDOW`         | overrides the architectural context window                                                    |
| `DIFFVAE_NA_BRICK`          | overrides the derived brick                                                                   |
| `DIFFVAE_NA_KV_CHUNK_TILES` | tiles per flash step; 8 = 256 tokens                                                          |




### Theirs


| variable                                     | does                                                     |
| -------------------------------------------- | -------------------------------------------------------- |
| `DIFFVAE_GNA=1`                              | take the stride from the picked block (`b == s`)         |
| `DIFFVAE_BLOCK=1`                            | 3D block-permuted Q                                      |
| `DIFFVAE_SP_FUSED=1`                         | the fused kernel instead of the streamed op              |
| `DIFFVAE_SP_TINNER`                          | make T the innermost flatten axis (default 1)            |
| `DIFFVAE_TP_PROJ`, `DIFFVAE_TP_HEADS`        | tensor-parallel over heads                               |
| `DIFFVAE_STAGES_WSP=1`                       | W-shard the deterministic stages too                     |
| `DIFFVAE_SLAB_FRAMES`                        | frame banding. **Off by default**; required at 6 s 1080p |
| `DIFFVAE_STAGE_TIMING`, `DIFFVAE_BLOCK_PROF` | the decode tree                                          |
| `DIFFVAE_NUM_LINKS`                          | CCL links                                                |


---



## 5. How a decode flows

```
latent
  └─ conv_in (denormalisation folded into the weights)
  └─ DeterministicStages           stages 1-4, NABlocks + upsamples
       └─ neighborhood_attention_3d(backend=...)     -> na3d.py
  └─ DiffVAEStage5.forward
       ├─ bands = _bands(t, DIFFVAE_SLAB_FRAMES, kernel)
       ├─ rope tables (factored: frame piece + time piece)
       └─ for band in bands:
            for block in 8 x DiffusionNABlock:
              context-inject -> AdaLN -> attention -> residual
                                  |
                                  ├─ "op_sp_w_sharded"     -> na3d.py, fused, block-permuted
                                  └─ "bricked_sp_w_sharded" -> neighborhood_attention.py
                                       ├─ neighbor_pad          (halo exchange)
                                       ├─ to_bricked            (natural -> bricked)
                                       ├─ neighborhood_sdpa     (our op)
                                       ├─ to_natural
                                       └─ slice off the halo
              -> AdaLN -> SwiGLU -> residual
  └─ unpatchify -> pixels
```

---



## 6. Reading order, if you are new to this

1. `neighborhood_window_rule.hpp` — 120 lines, and the whole technique is in it.
2. `neighborhood_plan.hpp` — the vocabulary.
3. `neighborhood_reference.py` — the same rules in torch, executable.
4. `test_neighborhood_sdpa.py` — what correct means, and which shapes break it.
5. `neighborhood_reader.cpp` — where geometry becomes memory traffic.
6. `na3d.py::_pick_block` — the reference's cost model, and its quality warning.
7. `diffvae_ltx_stage5.py::_build_rope_tables` — the constraint that shapes everything else.
