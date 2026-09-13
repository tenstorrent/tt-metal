# Neighborhood attention in the LTX-2.5 DiffVAE decoder

A map of every file involved in 3D neighborhood attention (NA) and its strided
generalization (GNA), across **two independent implementations** that live side by side:

- **the linear-order executor** — tokens in natural row-major order, a gather per query group and
dense masked SDPA over it (`neighborhood_attention_3d_linear_order`, plan in
`neighborhood_attention_plan.py`). Runs stage 1 and is the replicated oracle for the sharded tests.
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
| **gather / box**      | ours      | The union of the windows of **every** query in the chunk, rounded out to brick boundaries. What actually gets fetched. `gather_brick_count` in the plan, `gather=N tiles` in the log. |
| **slot**              | ours      | One key brick within the gather. The kernel's unit of K/V and of mask: it walks slots, not sites.                                                                             |
| **in-window slots**   | ours      | The slots one query **brick** can actually see — the window union of its own 32 queries. The remaining slots of the gather are `-inf` for that brick and still cost a mask tile and a matmul. |
| **narrowing**         | both      | Iterating a query brick over its in-window slots instead of the whole gather. The reference does it with `windowed_k_chunk_range()`; we do not.                               |

Two traps in the above, both of which have already cost a wrong estimate:

* **A query's window is not a brick's.** One query sees ~54 bricks; a brick of 32 queries sees the
  union of all 32, which for `brick=(2,8,2)` at window 11 is `7x3x7 = 147`. Sizing narrowing off
  the single-query figure overstates it by ~3x.
* **In-window slots do not shrink with the chunk.** 147 is a property of the brick and the window,
  not of the chunk it sits in. The *gather* grows with the chunk (147 at chunk `(1,1,1)`, 192 at
  `(2,1,2)`); the in-window count does not. Narrowing's headroom is the difference between them.

Note that **`sub-box` in the plan code means something else** — `neighborhood_plan.hpp` uses it for
the owned query region within the resident region, a sharding concept. Do not use it for the window
sense above.




### Cost model


| Term            | Meaning                                                                                                                                                                                                     |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **gather**      | The keys one query chunk must read: the union of its queries' context windows, rounded out to whole bricks.                                                                                                 |
| **box**         | Reference's name for the same thing.                                                                                                                                                                        |
| `box / vol`     | Keys gathered per query. Governs **gather traffic** only. Ours logs it as `keys/query`. It is **not** the score-tile count: with per-brick masks each query brick is scored against every gathered slot, so score tiles per query brick are `gather_brick_count` itself. A wider chunk lowers `keys/query` while *raising* score tiles. |
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



## 3. The plans



### `models/tt_dit/layers/neighborhood_attention_plan.py` (~1060 lines)

Everything that depends on the geometry (volume, window, stride, brick, mesh) and on no weights, for
both executors, built once per shape and cached. Nothing here imports `neighborhood_attention.py`.

The linear-order plan:

| symbol                                                          | is                                                            |
| --------------------------------------------------------------- | ------------------------------------------------------------- |
| `window_bounds(length, kernel, stride)` | per-site window `starts`/`ends`; a wrapper over `neighborhood_reference.context_window_origin`, so there is ONE window rule in Python |
| `plan_na3d`, `NA3DPlan`, `TileGroup`                            | group query tiles by window geometry; `NA3DPlan.describe()` prints the plan as prose |
| `na3d_torch` | the tiled torch reference: the same rule as `neighborhood_reference.py`, scaling to real volumes; the two are held equal in `test_neighborhood_reference.py` |
| `NA3DShard`, `NA3DGroup`, `NA3DDevicePlan`, `build_device_plan`, `cached_device_plan` | uploaded gather indices and masks, optionally split across the mesh |

The bricked plan:

| symbol                                  | does                                                                             |
| --------------------------------------- | -------------------------------------------------------------------------------- |
| `_choose_sharded_brick`, `brick_override` | the brick: exhaustive 32-site search minimising gathered bricks, or `DIFFVAE_NA_BRICK` |
| `_query_chunk_bricks(stride, brick)`    | `stride // brick` per axis where it divides, else 1. **Derived, never tuned.**   |
| `halo_sites(window, brick)`             | `ceil(window/2 / brick) * brick` — the halo, in whole bricks                     |
| `_tiles_per_kv_chunk(gather)`           | largest chunk that fits DST and divides the gather                               |
| `_build_relative_masks`, `_build_regime_masks` | the resident mask tables: relative at stride 1, per-regime under a GNA stride |
| `cached_bricked_plan(...)`              | the C++ planner per shard + uploaded tables, cached per geometry; unsharded is the one-shard case |

The sharded path builds **one plan per shard** and stacks the tables, uploading with
`mesh_axes=[sp_axis]` so each device reads its own origins. It asserts the plans agree on
`chunk_count`, `gather_brick_count`, `gather_bricks`, `volume_chunks` — they must, because one
program serves the mesh.


### `models/tt_dit/layers/block_permute.py` -- RETIRED 2026-09-10

Was their equivalent of our bricking (reorder tokens so a block is contiguous). Deleted along
with `na3d.py::_pick_block`, `DIFFVAE_BLOCK` / `DIFFVAE_GNA` and the SDPA op's `neighborhood_block`
argument: at the production shard widths no legal block ever existed, so the reference executor
always ran its strided mode. See `PLAN_retire_block_permute.md`. The strided executor itself
followed on 2026-09-11.

### `models/tt_dit/utils/decode_tree.py` (~275 lines)

The timing tree. `open_span` / `close_span`, `Node`, `roots()`, `render()`. Spans nest by a
thread-local stack; siblings with the same label are pooled with an `n` count.
`ENABLED` ← `DIFFVAE_STAGE_TIMING`, `DEEP` ← `DIFFVAE_BLOCK_PROF`, `LIVE` ← `DIFFVAE_STAGE_LOG`
(one stdout line per span open/close while the decode runs, so a hang shows as the last `>`
with no `<`; the tree itself is unchanged).
**Not valid under trace capture**, and absolute totals are inflated by one
`synchronize_device` per span open/close.

### Their SDPA kernels

The general SDPA op's neighborhood mode (`neighborhood_3d`, `neighborhood_w_shard`,
`neighborhood_gather`, `neighborhood_stride`; its additions to `windowed_loop_geometry.hpp`,
`windowed_mask_gen.hpp`, `reader_interleaved.cpp`, `writer_interleaved.cpp` and the shared factory,
plus `neighborhood_gather.hpp`) was removed on 2026-09-11, after its last executor went. Those files
are back at their upstream content. The one thing our op still shares with the general op's kernels
is `compute_common.hpp::matmul_blocks`, whose `mask_subblock_stride` parameter (default 0, the
upstream behaviour) is what lets the per-brick mask advance down the query rows.

---



## 4. Environment variables



### Ours


| variable                    | does                                                                                          |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| `DIFFVAE_STAGE5_BACKEND`    | selects the stage-5 executor: `bricked_sp_w_sharded` (default), `bricked` (replicated) or `linear_order` (replicated) |
| `DIFFVAE_DET_NA3D_BACKEND`  | same choice for the deterministic stages 1–4 only — **separate knob**, does not reach stage 5  |
| `DIFFVAE_STAGES_BACKEND`    | the executor for the W-sharded deterministic stages 1–3 when `DIFFVAE_STAGES_WSP=1`; `bricked_sp_w_sharded` (default) is the only W-sharded one left. Read by the pipeline adapter and `test_decode_timing.py` |
| `DIFFVAE_S5_GNA_STRIDE`     | stage-5 stride, physical `(t,h,w)`; read only by `DiffVAEStage5Config.resolved_gna_stride`, which feeds every stage-5 backend. An explicit `gna_stride=` on the config wins over it. |
| `DIFFVAE_NA_WINDOW`         | overrides the architectural context window                                                    |
| `DIFFVAE_NA_BRICK`          | overrides the derived brick: `bt,bh,bw` for every volume, or keyed by full volume `T,H,W:bt,bh,bw;...` so one stage can be forced without moving the others |
| `DIFFVAE_NA_KV_CHUNK_TILES` | tiles per flash step; 8 = 256 tokens                                                          |


### Ours — diagnostics

All default off. The wrong-output probes of the 2026-09-10 mask investigation (`SKIP_KV`,
`MASK_MEMSET_ONLY`, `TABLE_ALWAYS`) and the `PER_BRICK_MASK` override were removed on 2026-09-11
once that investigation closed; the op reads only `DIFFVAE_NA_UNSAFE_CHUNK` from the environment.

| variable                       | does                                                                     |
| ------------------------------ | ------------------------------------------------------------------------ |
| `DIFFVAE_NA_CHUNK_BRICKS`      | force the query chunk, in BRICKS (`t,h,w`)                               |
| `DIFFVAE_NA_UNSAFE_CHUNK`      | lift the plan's `chunk == stride` check, needed with the above at stride 1. The factory then switches to per-brick masks on its own — see `neighborhood_plan.cpp` |
| `DIFFVAE_NA_HALO_TOPOLOGY`     | `ring` retries the halo on ring — see the deadlock note in `_halo_exchange` |
| `DIFFVAE_NA_HALO_LINKS`        | halo link count only                                                     |
| `DIFFVAE_NA_HALO_PERSISTENT`   | halo without the persistent buffer                                       |
| `DIFFVAE_EXCLUSIVE`            | restore the DiffVAE's evict-the-DiT residency behaviour                  |




### Shared decode knobs


| variable                                     | does                                                     |
| -------------------------------------------- | -------------------------------------------------------- |
| `DIFFVAE_TP_PROJ`, `DIFFVAE_TP_HEADS`        | tensor-parallel over heads                               |
| `DIFFVAE_STAGES_WSP=1`                       | W-shard the deterministic stages too                     |
| `DIFFVAE_SLAB_FRAMES`                        | frame banding. **Off by default**; required at 6 s 1080p |
| `DIFFVAE_STAGE_TIMING`, `DIFFVAE_BLOCK_PROF`, `DIFFVAE_STAGE_LOG` | the decode tree, and its live progress lines |
| `DIFFVAE_NUM_LINKS`                          | CCL links                                                |



### Known constraints

* `neighbor_pad_async` deadlocks on `Topology.Ring`. `_halo_exchange` pins that one call to Linear
  while everything else still runs ring; its docstring records what was ruled out (channel width,
  link count, persistent buffer).
* Exact NA at 6 s 1080p does not fit co-resident with the DiT. Either band harder
  (`DIFFVAE_SLAB_FRAMES=48`) or fall back to exclusive residency (`DIFFVAE_EXCLUSIVE=1`). The
  decode-only timing test runs fine at the default banding because nothing else is resident.


---



## 5. How a decode flows

```
latent
  └─ conv_in (denormalisation folded into the weights)
  └─ DeterministicStages           stages 1-4, NABlocks + upsamples
       └─ neighborhood_attention_3d_linear_order  -> neighborhood_attention.py (plan: neighborhood_attention_plan.py)
  └─ DiffVAEStage5.forward
       ├─ bands = _bands(t, DIFFVAE_SLAB_FRAMES, kernel)
       ├─ rope tables (factored: frame piece + time piece)
       └─ for band in bands:
            for block in 8 x DiffusionNABlock:
              context-inject -> AdaLN -> attention -> residual
                                  |
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
6. `diffvae_ltx_stage5.py::_build_rope_tables` — the constraint that shapes everything else.
