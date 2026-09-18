# Neighborhood attention in the LTX-2.5 DiffVAE decoder

A map of every file involved in 3D neighborhood attention (NA) and its strided
generalization (GNA), across **two independent implementations** that live side by side:

- **the linear-order executor** — tokens in natural row-major order, a gather per query group and
dense masked SDPA over it (`neighborhood_attention_3d_linear_order`, plan in
`neighborhood_attention_plan.py`). Runs stage 1 and is the replicated oracle for the sharded tests.
- **ours** — a self-contained `neighborhood_sdpa` op with all geometry in one host-testable
file. Runs the deterministic stages 2-4 and stage 5 (the bricked executor), selected by backend
name.

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
| **band / slab**       | reference | A range of frames processed together to bound peak memory. `DiffVAEOptions.slab_frames`.                                                                                                |
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

#### `ttnn/.../sdpa/device/neighborhood_plan.hpp` (~180 lines)

The vocabulary. Everything else imports its nouns from here; the point and shape types themselves
come from `kernels/neighborhood_point3.hpp` (section 2.4), which host and kernels share.

```cpp
constexpr uint32_t SITES_PER_BRICK = 32;
enum class Regime { Low, Interior, High };

// from neighborhood_point3.hpp: Point3<Scalar, Unit> and Shape<MeasuredIn, Per>
using Site         = Point3<uint32_t, Unit::Sites>;   // a POSITION, in sites
using SiteOffset   = Point3<int32_t,  Unit::Sites>;   // a SIGNED position -- see sharding
using BrickPoint   = Point3<uint32_t, Unit::Bricks>;
using ShapeInSites = Shape<Unit::Sites>;              // a SIZE, in sites
using ShapeInBricks = Shape<Unit::Bricks>;
using BrickShapeInSites = Shape<Unit::Sites, Unit::Bricks>;   // sites per brick
```

A size and a position are distinct types, and so are the units, so a brick grid cannot be passed
where a site region belongs and a chunk index cannot be mistaken for a brick index.

`NeighborhoodConfig` — the complete description of one problem:


| field                | meaning                                                            |
| -------------------- | ------------------------------------------------------------------ |
| `volume`             | the **global** token grid                                          |
| `context_window`     | what one query group attends to                                    |
| `stride`             | query group extent                                                 |
| `brick`              | layout unit; `brick.count() == 32`                                 |
| `query_chunk_bricks` | how many bricks share one gather                                   |
| `shard_extent`       | what this device holds (owned + halo); zero means "same as volume" |
| `shard_origin`       | **signed** — where that sits in the global volume                  |
| `query_extent`, `query_origin` | the sub-region of the resident tensor this device produces output for (Q and the output are sized by it; K, V and the gather by the resident extent). Zero means "all of it" |


`NeighborhoodPlan` — what `build_plan` produces, cached because it uploads index tables:


| field                                           | meaning                                                                    |
| ----------------------------------------------- | -------------------------------------------------------------------------- |
| `volume_bricks`, `brick_count`                  | the **resident** region measured in bricks                                 |
| `query_bricks`, `query_brick_count`, `query_origin_bricks` | the query region measured in bricks, and where it starts       |
| `volume_chunks`, `chunk_count`                  | the query region measured in **work items**                                |
| `gather_extent`, `gather_sites`, `gather_tiles` | site-exact gather                                                          |
| `gather_bricks`, `gather_brick_count`           | rounded out to whole bricks — what a tile-granular read can actually fetch |
| `gather_origin_by_chunk`                        | one origin per chunk, rounded **down** to a brick boundary                 |




#### `neighborhood_plan.cpp` (~350 lines)


| function                                          | does                                                                                                                                                                                                          |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `choose_brick(window)`                            | Enumerates the power-of-two factorisations of 32, returns the one minimising `∏(window_axis + brick_axis − 1)`, ties broken toward the most cubic brick. **A function, not a constant**: an 11³ window wants `(2,4,4)`; a `(1,11,11)` window wants `(1,4,8)`. Exposed as `ttnn.transformer.neighborhood_choose_brick`; the sharded planner wraps it in `_choose_sharded_brick` (section 3). |
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
struct NeighborhoodExtents {
    BrickShapeInSites brick_sites;   // sites per brick -- a conversion factor, not a region
    ShapeInSites context_window, stride, volume, query_chunk, resident;
    SiteOffset shard_origin;         // signed
};
```

One struct rather than nine loose arguments, so the mask generator's caller cannot transpose
height and width without noticing.



#### `kernels/neighborhood_chunk_layout.hpp` (~90 lines)

Small named helpers shared by reader and writer, so the two cannot decode a chunk differently:
`linear_to_point3` / `point3_to_linear` (row-major over `(time, height, width)`, applied once at
chunk level and once at brick level, with the unit named at the call site), the chunk's first brick
and its `index`-th brick, `point3_is_inside` (a brick hanging past the volume holds only ghost
sites) and `tile_offset` (where one `(brick, head)` pair's tiles begin in the **site-major**
`[batch, bricks * 32, heads, head_dim]` layout).

### 2.3 The op — ttnn plumbing, no geometry

| file | lines | holds |
| --- | --- | --- |
| `device/neighborhood_sdpa_device_operation_types.hpp` | ~60 | `NeighborhoodSDPAParams` and `NeighborhoodSDPAInputs` |
| `device/neighborhood_sdpa_device_operation.{hpp,cpp}` | ~280 | validation, output spec, the explicit program hash |
| `device/neighborhood_sdpa_program_factory.cpp` | ~350 | work split, circular buffers, compile args, kernel launch |
| `neighborhood_sdpa_nanobind.{cpp,hpp}` | ~280 | the three Python entry points |

`NeighborhoodSDPAParams`: a `NeighborhoodConfig`, `head_count` (cannot be read off the shape: Q is
`[batch, 1, bricked_sites, heads * head_dim]` so that a TILE row is 32 sites), `scale` (1.0; LTX
pre-scales Q), `tiles_per_kv_chunk` (key bricks per flash step) and the compute / memory configs.
`NeighborhoodSDPAInputs`: Q, K, V in TILE layout and bricked order; the uint32 `gather_origin_table`,
one row per chunk, **uploaded rather than computed in the kernel** so the reader and the host planner
cannot disagree about geometry; an optional `interior_mask` (the uploaded mask table of section 3);
an optional preallocated output.

Validation runs `validate_config` first, so a bad geometry fails as geometry and not as a tensor
shape. K and V must match each other; Q need not match them, since with a query sub-region Q spans
only the owned sites while K and V span owned plus halo. Site counts are checked against the plan's
brick counts, not against the volume, because the permute pads to whole bricks.
`tiles_per_kv_chunk` is bounded twice: by whole bricks, and by DST capacity (8 tiles, half that
under fp32 accumulation), because a chunk wider than DST returns wrong numbers rather than faulting.
The program hash is explicit: the eight stage-5 blocks must share one compiled program, and the
origin table's contents are a runtime buffer.

The factory makes one work item per `(batch, head, query chunk)`, bricks varying fastest so a core's
items are spatially adjacent. It picks the mask mode: a chunk wider than the stride needs a mask per
brick (reachable only under `DIFFVAE_NA_UNSAFE_CHUNK`); otherwise one tile per gather slot broadcasts
down the chunk. `cb_mask` is sized to a whole work item so its pages cycle back to the same L1
addresses every item, which is what lets the reader skip rewriting them for a run of unclamped
bricks; that size must agree with `interior_table_supported` in the reader. `cb_resident_mask` holds
one regime's whole uploaded mask set. `subblock_h` is always 1 (`matmul_blocks` re-reads the mask
from the CB front per in0 subblock, which is only right when each subblock is one query row).
Compile-time arguments are written by name from the `kernel_args` enums; `REDUCE_OP`, `REDUCE_DIM`
and `EXP_APPROX_MODE` go in as defines because `compute_common.hpp` reads them as macros.

The binding exposes `ttnn.transformer.neighborhood_choose_brick(context_window)`,
`neighborhood_plan(volume, context_window, stride, brick, query_chunk_bricks=, shard_extent=,
shard_origin=, query_extent=, query_origin=)` (a dict: `brick_count`, `query_bricks`, `chunk_count`,
`volume_chunks`, `gather_extent` / `gather_sites` / `gather_tiles` / `gather_bricks` /
`gather_brick_count`, the flattened `gather_origin_table` and its `gather_origin_columns`) and
`neighborhood_scaled_dot_product_attention(q, k, v, gather_origin_table, interior_mask=, <the same
geometry>, head_count, scale=1.0, tiles_per_kv_chunk=8, memory_config=, compute_kernel_config=)`.

### 2.4 Kernels

#### `kernels/dataflow/neighborhood_reader.cpp` (~900 lines)

The only kernel that knows what a context window is. Per work item it feeds the Q tiles of the
chunk's bricks, then the gather's K and V tiles `tiles_per_kv_chunk` bricks at a time with a
matching additive mask. The chunk is what makes this affordable: its bricks form one query group,
so the gather happens once per chunk and the mask is one tile per gather slot.

Where a mask tile comes from, in order of preference: the uploaded **relative** table at stride 1
(indexed by `key_brick - query_brick` per axis; `relative_span_low/high` and
`relative_table_index` mirror `relative_mask_span` and `_build_relative_masks` in the plan module
and must match them to the tile); the uploaded **per-regime** set under a GNA stride (`chunk_regime`
picks one of the 27 clamp classes); or **generated on device** for the bricks that straddle a
clamp transition, at most one per edge per axis. A brick's mask depends on its position only
through clamping, which is the same collapse the linear-order plan gets by grouping query tiles by
window geometry. Persistence (skipping the rewrite when the CB pages cycle, and keeping per-brick
blocks keyed on the chunk's clamp signature) is described in `kernels/MASK_PERSISTENCE.md`.

#### `kernels/dataflow/neighborhood_mask_gen.hpp` (~280 lines)

The device-side mask generator, under `dataflow/` on purpose: the compute kernel never sees a
window. `to_global_site` (a low-edge halo device sits at a negative origin), `classify_brick` into
`BrickCoverage {AllVisible, NoneVisible, Mixed}` so uniform bricks are a constant fill, and
`fill_mask_tile` for the mixed ones, which knows that a bfloat16 tile is four 16x16 faces and not
`row * 32 + column`. Its window rule is `neighborhood_window_rule.hpp`, the same header the host
planner includes.

#### `kernels/compute/neighborhood_sdpa.cpp` (~180 lines)

Flash attention over one query chunk. **Readability invariant: no neighborhood concepts** -- it
receives query tile rows, a stream of K/V tiles and an additive mask, and runs online softmax.
A chunk is a whole number of tile rows, so `query_tile_rows` is both the matmul's M and its in0
subblock count and `subblock_h` is always 1. `mask_subblock_stride` 0 broadcasts one mask down the
chunk; `tiles_per_kv_chunk` selects one mask per brick. The shared `compute_common.hpp::matmul_blocks`
is the only code this op shares with the general SDPA kernels.

#### `kernels/dataflow/neighborhood_writer.cpp` (~120 lines)

Drains one query brick's normalised output per work item, **in bricked order**, so the next block
consumes it without a permute. Builds the three constant tiles once (the reduce identity, a genuine
zero tile that `matmul_blocks` folds the mask through, and ones down column 0 for the deferred
row-sum) and skips bricks that hang past the volume.

#### `kernels/neighborhood_point3.hpp` (~170 lines)

`Point3<Scalar, Unit>` and `Shape<MeasuredIn, Per>` with `Unit {Sites, Bricks, Chunks}` tags, and the
aliases section 2.1 uses. It replaced seven earlier spellings of "three numbers" (`Site`, `Offset3`,
`SignedAxisOffsets`, `SiteInBrick`, `BrickCoordinate`, `Extent3`, `AxisExtents`) so a unit mismatch
is a compile error. Header-only, no includes beyond `<cstdint>` and `<array>`, included by the host
planner and every kernel.

### 2.5 Python

#### `models/tt_dit/layers/neighborhood_permute.py` (~220 lines)

Natural <-> bricked token order, ROW_MAJOR only: `padded_volume`, `brick_grid`, `brick_count`,
`sites_per_t_brick`, `to_bricked_grid` / `to_bricked` and `from_bricked_grid` / `to_natural`.
Bricks tile the volume time-major, the same order `neighborhood_chunk_layout.hpp` decodes. Applied
once at stage entry and exit on the keep-bricked path, per block on the deterministic stages.

#### `models/tt_dit/layers/neighborhood_reference.py` (~190 lines)

The definition of correct: dense masked attention over the whole volume, test-sized only.
`context_window_origin` is **the** Python window rule (the plan module's `window_bounds` and
`_window_origin` are wrappers over it); `snap_extent`, `validate` (rejects what the device op
rejects), `neighborhood_mask`, `neighborhood_attention_3d`.

#### `models/tt_dit/layers/neighborhood_attention.py` (~620 lines)

The three executors, one contract: `q`/`k`/`v` are `(B, T, H, W, heads, head_dim)`, normed, RoPE'd
and Q pre-scaled; the return is `(B, T, H, W, heads * head_dim)` ROW_MAJOR.

- `neighborhood_attention_3d_linear_order` -- natural order, `ttnn.embedding` gather per query
  group, dense masked SDPA, optionally query-sharded across the mesh with K/V replicated. Stage 1
  and the oracle. Refuses a non-trivial `gna_stride`.
- `neighborhood_attention_3d_bricked` -- the whole volume on every chip: `to_bricked`, the op,
  `to_natural`. `DIFFVAE_NA_WINDOW` is read here.
- `neighborhood_attention_3d_bricked_w_sharded` -- this chip's W-shard: `as_volume` (the flat
  head-major form `nlp_create_qkv_heads` emits is transposed to site-major), brick, halo-exchange K
  and V by whole bricks on `W_br` through `_halo_exchange` (`neighbor_pad_async` pinned to Linear,
  sticks split by `_halo_split` when they exceed what the op moves intact), the op with
  `query_extent` / `query_origin` so Q and the output address the owned grid while K, V and the
  gather address the resident one, `to_natural` unless `already_bricked`, then the head all-gather
  over `tp_axis`. `_compute_kernel_config` is HiFi2 with an exact exp, matching the general SDPA op
  the oracle runs; `_tp_trace` is the hang locator.
- `NAKernel` / `NA_KERNELS` / `resolve_na_kernel` -- the backend registry: a name
  (`linear_order`, `bricked`, `bricked_sp_w_sharded`) and the layout decisions that follow from it
  (`w_sharded`, `bricked`, `keep_bricked`). Both the deterministic stages and stage 5 resolve their
  `DiffVAEOptions` backend string here, once, and hand the record down.
- `neighborhood_attention_3d` -- the dispatcher: runs whichever executor the `NAKernel` names, with
  the arguments that executor understands. The only call site of the three executors in the model.

### 2.6 Tests


| file                                                      | covers                           |
| --------------------------------------------------------- | -------------------------------- |
| `tests/ttnn/unit_tests/gtests/test_neighborhood_plan.cpp` | geometry vs a brute-force oracle, 11 cases (not in the default build graph; compile it standalone) |
| `models/tt_dit/tests/unit/test_neighborhood_permute.py`   | bricked↔natural round-trip, the index formula, W_br padding and T_br slicing against pad/slice-then-brick |
| `models/tt_dit/tests/unit/test_neighborhood_reference.py` | the torch reference itself: window rule vs NATTEN at every stride, same-count / inside-own-window / full-attention invariants, `window_bounds` and `na3d_torch` held equal to it |
| `models/tt_dit/tests/unit/test_neighborhood_sdpa.py`      | the op vs torch — 44 cases       |
| `models/tt_dit/tests/unit/test_neighborhood_sdpa_perf.py` | scale timing, no correctness     |
| `models/tt_dit/tests/unit/test_neighborhood_linear_order.py` | the linear-order executor vs `na3d_torch`, replicated and mesh-sharded; `NA3DPlan.describe()` accounts for every tile |
| `models/tt_dit/tests/unit/test_neighborhood_bricked_w_sharded.py` | the W-sharded bricked executor vs the host reference on the 4x8 mesh, heads presharded, volume and flat forms |
| `models/tt_dit/tests/models/vae/test_diffvae_rope.py`     | the three RoPE encodings vs one torch oracle and each other, lane level, float32 |
| `models/tt_dit/tests/models/vae/test_diffvae_ltx.py`      | NABlock, its DetBlockOptions arms, DeterministicStages and DiffVAEDecoder: gates vs upstream captures, bricked vs replicated, timing instruments |
| `models/tt_dit/tests/models/vae/test_diffvae_ltx_stage5.py` | stage 5 vs ltx_core: RoPE, parity replicated / sharded / bricked, production-width, band geometry |


The op test's parametrisation is where the coverage lives:

- `stride_one`, `stride_equals_brick`, `partial_gather`, `many_items_per_core`,
`multi_brick_chunk`, `chunk_overhangs_volume`
- `widest_chunk` / `narrow_chunk` — whether the online rescale runs
- `one_tile_row` **/** `two_tile_row` — `head_dim` 32 vs 64. Added after the K-layout bug.
- `test_shards_match_the_whole_volume` — two shards, different origins, same program
- `test_symmetric_halo_shards_match_the_whole_volume` — three shards including a **negative** origin,
  with an even brick and with the width-1 brick the odd shard widths use
- `test_interior_table_matches_generated_masks` / `test_interior_table_per_brick_persistence` — the
  uploaded relative table against the device generator, and the persistent per-brick block
- `test_choose_sharded_brick_*` — the production and deterministic-stage brick choices pinned, the
  stride > 1 delegation, and the oversized-brick rejection

---



## 3. The plans



### `models/tt_dit/layers/neighborhood_attention_plan.py` (~940 lines)

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


### The retired block-permute path

`models/tt_dit/layers/block_permute.py` was their equivalent of our bricking (reorder tokens so a
block is contiguous), and `neighborhood_permute.py` implemented the same 3-D permutation a second
time. Rather than unify the two, the older one was retired end to end, stages 2-5 moving onto the
bricked executor:

- 2026-09-10: `block_permute.py`, `na3d.py::_pick_block`, `DIFFVAE_BLOCK` / `DIFFVAE_GNA` and the
  general SDPA op's `neighborhood_block` argument. At the production shard widths (W_local 15 and
  30 at 1080p) no legal block ever existed, so the reference executor always ran its strided mode;
  block order was never live in production.
- 2026-09-11: the strided executor `neighborhood_attention_3d_op_sp_w_sharded` (kernel name
  `op_sp_w_sharded`: full-W K/V all-gather + `wrow` retile into the general SDPA op) and, once no
  executor was left, the general op's C++ neighborhood mode (see "Their SDPA kernels" below).
- 2026-09-13: `na3d.py` itself; its gather backend became `neighborhood_attention_3d_linear_order`
  and its planner moved to `neighborhood_attention_plan.py`.

Naming: the deleted path is the **block-permute executor**, the surviving one the **bricked
executor**. Neither is "fused" -- both fuse the gather into the kernel, and the word already names
the fused RoPE / qkv / SwiGLU forms and the `fused-sdpa` timing-tree row.

Decisions taken on the way, still in force:

- **TP stays over heads for the deterministic stages.** The bricked executor reassembles heads with
  a real site-major -> head-major permute before the head all-gather when a chip holds more than
  one head. Dropping TP was rejected: up to 4x attention compute per chip to save a 10-line change.
- **Shard widths that are not brick-aligned (stages 2 and 3) are handled by brick width 1**, found
  by `_choose_sharded_brick`'s odd-width search. The alternative, swapping the SP and TP mesh axes
  for the deterministic stages, was priced with `DiffVAEOptions.stages_sp_axis` and rejected: it costs
  2.2 s per decode because stage 5 cannot follow (4 heads do not TP 8 ways), so the W-sharded
  deterministic -> stage-5 context handoff becomes a gather and reshard.
- Stage 1 (index 0) stays on the replicated linear-order backend: W=60 does not divide the size-8
  mesh axis.

The gate baseline (`tests/models/vae/diffvae_gate_baseline.json`) carries all 20 gates, bricked
included. A full `run_diffvae_gates.sh` takes about 15 min: the w480_h272 production-width row
alone is ~8 min, most of it the ltx_core reference on the host, and it carries its own 1800 s
pytest timeout. On the device broker, whose jobs cap at 1500 s, split the run with `-k` and merge
the ledgers before `diffvae_gate_compare.py --record`.

Open, optional speed work: running the deterministic stages keep-bricked (convert to bricked
order once at stage entry and back at exit, as stage 5 does). Stage 1, replicated on the linear-order
executor, is ~525 ms of the decode now that the production options set `det.fused_qkv` (~630 ms
under block profiling, ~1080 ms before with three separate projections and the unfused rotation;
that stage has no TP axis, so the column-parallel flag the other stages get fused qkv from never
reached it). The production pipeline's VAE decode is 12.00 s with it, from 12.2-12.35 s.

One thing that bit during the migration and will again: kernel sources are JIT-only, so the host
syntax check never sees them. Run the JIT compile command with `-fsyntax-only` after a kernel edit,
before touching the device.

### `models/tt_dit/utils/timing_tree.py` (~360 lines)

The timing module. `span(device, label, category=, root=, deep=)` is the one primitive: sync,
open a node, body, sync, close with the elapsed ms; the same object is a decorator for a method
whose whole body is one span (`@span("mesh_device", label, ...)`, label and device may be callables). Spans nest by one module-level stack (the decode is single-threaded); siblings with the same
label are pooled with an `n` count. `open_span` / `close_span`, `Node`, `roots()`, `render()`.
`ENABLED` ← `TT_DIT_STAGE_TIMING`, `DEEP` ← `TT_DIT_BLOCK_PROF` (the `deep=True` spans inside
the blocks), `LIVE` ← `TT_DIT_STAGE_LOG` (one stdout line per span open/close while the decode
runs, so a hang shows as the last `>` with no `<`; the tree itself is unchanged).
`TT_DIT_TREE_DEPTH` caps the rendered depth. The DiffVAE modules and the executors call
`timing_tree.span` directly; no timing helper lives in the models.
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



## 4. Configuration

### Decoder options (constructor arguments)

How the decoder runs is a `DiffVAEOptions` (in `diffvae_ltx.py`), built once by whoever
constructs the decoder and handed down; nothing below `DiffVAEDecoder.__init__` reads the
environment. `DiffVAEOptions.production()` is what the runner scripts ship. The pipeline takes it
as `create_pipeline(..., diffusion_decoder=True, diffvae_options=...)`; the pytest entry points
build it from the `--diffvae-*` options (`pytest --help`, group "LTX-2.5 DiffVAE"), and
`time_module` from its own flags.

| field                        | does                                                                                          |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| `stage5_backend`, `stage5_sp_axis`, `stage5_tp_axis` | the stage-5 executor (`linear_order` replicated, `bricked` replicated, `bricked_sp_w_sharded`), its shard axis and the TP-over-heads axis |
| `stages_backend`, `stages_sp_axis`, `stages_tp_axis` | the same for the deterministic stages 1–3 — **separate choice**, does not reach stage 5; stage 1 always runs replicated |
| `det` (a `DetBlockOptions`)  | the deterministic block forms: `fused_qkv`, `colpar_qkv`, `fused_rope`, `fused_swiglu`, `tp_mlp`. `resolve(tp_axis)` applies the implications (colpar implies fused qkv, tp_mlp implies fused swiglu, both need a TP axis) |
| `stage5_fused_qkv`, `stage5_tp_proj` | the stage-5 projection forms; `tp_proj` is column-parallel qkv over `stage5_tp_axis`, on by default |
| `gna_stride`                 | stage-5 stride, physical `(t,h,w)`, feeding every stage-5 backend                             |
| `slab_frames`                | frame banding. **Off by default**; required at 6 s 1080p                                       |
| `device_boundaries`          | the decode's two host boundaries run on device (ghost pad and flatten in; noise, pad trim and depth-to-space out). Needed by the traced decode and by `yuv` output |
| `exclusive_residency`        | evict the DiT before decoding; None means "unless stage 5 is sharded"                          |

### Environment variables — ours

| variable                    | does                                                                                          |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| `DIFFVAE_NA_WINDOW`         | overrides the architectural context window                                                    |
| `DIFFVAE_NA_BRICK`          | overrides the derived brick: `bt,bh,bw` for every volume, or keyed by full volume `T,H,W:bt,bh,bw;...` so one stage can be forced without moving the others |
| `DIFFVAE_NA_KV_CHUNK_TILES` | tiles per flash step; 8 = 256 tokens                                                          |
| `DIFFVAE_NA_FIDELITY`, `DIFFVAE_NA_APPROX_EXP` | A/B knobs on the op's compute config; the default is HiFi2 with an exact exp |


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
| `DIFFVAE_TP_TRACE`             | sync and log around every step of the sharded executor, so a hang names the op |
| `DIFFVAE_AG_CHUNKS`, `DIFFVAE_AG_WORKERS`, `DIFFVAE_AG_BUFS` | all-gather hyperparameters in `parallel/manager.py` |
| `DIFFVAE_MEM_LOG`              | the DRAM / CCL-cache probes in `utils/memory_log.py`                     |




### Shared decode knobs


| variable                                     | does                                                     |
| -------------------------------------------- | -------------------------------------------------------- |
| `TT_DIT_STAGE_TIMING`, `TT_DIT_BLOCK_PROF`, `TT_DIT_STAGE_LOG` | the timing tree, and its live progress lines |
| `TT_DIT_TREE_DEPTH`, `TT_DIT_TREE_OUT`, `TT_DIT_TREE_ALL` | rendered depth, a file to write the trees to, and rendering the warm-up passes too |
| `DIFFVAE_CHECKPOINT`, `DIFFVAE_CAPTURE`, `LTX_CORE_SRC` | where the standalone tests find the weights, the upstream capture and ltx_core |



### Known constraints

* `neighbor_pad_async` deadlocks on `Topology.Ring`. `_halo_exchange` pins that one call to Linear
  while everything else still runs ring; `DIFFVAE_NA_HALO_TOPOLOGY=ring` is the retest switch.
* Exact NA at 6 s 1080p does not fit co-resident with the DiT. Either band harder
  (`slab_frames=48`) or fall back to exclusive residency (`exclusive_residency=True`). The
  decode-only timing test runs fine at the default banding because nothing else is resident.
* The `DetBlockOptions` forms and `stage5_fused_qkv` change which parameters a block owns (one fused
  `qkv` or three projections; packed `gate_up` or `w_gate` + `w_up`), and the weight cache is keyed by
  parallel config, mesh and dtype alone. `DiffVAEDecoder.parameter_layout()` therefore names the cache
  subfolder (`diffvae/det-q1m1-q1m1-q1m1-q1m1_s5-q3m1-<hash>/...`): the forms read off the built
  modules, then a hash of every parameter's name and shape. Change a form, or rename a parameter, and
  the first run writes a fresh cache under the new token; the old directory is left behind.


---



## 5. How a decode flows

```
latent
  └─ conv_in (denormalisation folded into the weights)
  └─ DeterministicStages           stages 1-4, NABlocks + upsamples
       ├─ stage 1: replicated, neighborhood_attention_3d_linear_order
       └─ stages 2-4: W-sharded, "bricked_sp_w_sharded" PER CALL -- every block does
            to_bricked -> halo exchange -> neighborhood_sdpa -> to_natural on its own
  └─ DiffVAEStage5.forward          KEEP-BRICKED: the volume is converted once, not per block
       ├─ bands = _bands(t, slab_frames, kernel)
       ├─ brick x + context           (_brick_activation, once per band; RoPE tables built bricked)
       ├─ rope tables (factored: frame piece + time piece)
       └─ for block in 8 x DiffusionNABlock, for band in bands:
              context-inject -> AdaLN -> attention -> residual
                                  |
                                  └─ "bricked_sp_w_sharded", already_bricked=True
                                       ├─ neighbor_pad on W_br    (halo exchange, whole bricks)
                                       ├─ neighborhood_sdpa       (our op)
                                       └─ head all-gather         (TP; stays bricked, no to_natural)
              -> AdaLN -> SwiGLU -> residual
       ├─ norm_out -> conv_out        (still bricked)
       └─ unbrick                     (_unbrick_activation, once per band, ghosts cropped)
  └─ unpatchify -> pixels
```

Keep-bricked is the `keep_bricked` flag on `NAKernel`: the stage converts its activation to
bricked site order at entry and back at exit, so the eight blocks and the executor
(`already_bricked=True`) never pay the 7-D permute. The deterministic stages do not do this yet;
their per-block `to_bricked` / `to_natural` is the open item in section 3.

---



## 6. Reading order, if you are new to this

1. `neighborhood_window_rule.hpp` — 120 lines, and the whole technique is in it.
2. `neighborhood_plan.hpp` — the vocabulary.
3. `neighborhood_reference.py` — the same rules in torch, executable.
4. `test_neighborhood_sdpa.py` — what correct means, and which shapes break it.
5. `neighborhood_reader.cpp` — where geometry becomes memory traffic.
6. `diffvae_ltx_stage5.py::_build_rope_tables` — the constraint that shapes everything else. The
   angles under it, and under the deterministic stages' half-width tables, come from one place:
   `models/vae/diffvae_rope.py` (dim split, inverse frequencies, lane writer, the two lane-order matrices).
