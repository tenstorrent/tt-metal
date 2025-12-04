
# Tenstorrent `tt-metal`: Integral Image (Summed-Area Table) Kernels — High-Level Guide (Axis Spec: **[B, W, H, C]**)

# 🧩 Overview

An **integral image** (also called a *summed-area table*) is a data structure that stores the cumulative sum of pixel values up to each coordinate in a 2D image.
Formally, each pixel at position *(y, x)* contains the sum of all pixels above and to the left of *(y, x)*, inclusive.
👉 [Learn more on Wikipedia](https://en.wikipedia.org/wiki/Summed-area_table)

Integral images are widely used as a **preprocessing step** in image processing and computer vision algorithms, such as fast local averaging, feature extraction, and object detection (e.g., in the original Viola–Jones face detector).

In this kernel, we extend the concept to **multi-channel and batched tensors**, enabling parallel computation of integral images across multiple channels and images simultaneously.

Conceptually, this operation is equivalent to performing two cumulative sums:
`ttnn.cumsum(ttnn.cumsum(x, dim=-2), dim=-3)` (on a compliant tensor, which is described right below)
However, our implementation fuses both passes into a single optimized kernel to **reduce memory transfers** and **improve performance** on Tenstorrent hardware.

> **Axis convention for this document:** The input tensor is laid out as **[Batches, Width, Height, Channels]** — abbreviated **[B, W, H, C]**. Only **B=1** is supported by the current implementation/tests.
> **Mapping to kernel naming:** The code uses function names like `cumsum_cube_axis_2` and `cumsum_cube_axis_3`. In this guide we will always speak in terms of **Width (W)** and **Height (H)** to avoid confusion:
>
> - Kernel’s **“axis 2”** ≙ **Width (W)** scan (left→right).
> - Kernel’s **“axis 3”** ≙ **Height (H)** scan (top→bottom).
>
> Channels (C) are sharded across cores.

---

## 0) TL;DR — mental model (with **[B, W, H, C]**)

- Do a **W (left→right) scan** per block, **save** the block’s right edge.
- Add that right edge to the **next block** along **W**.
- Do an **H (top→bottom) scan** inside each tile.
- Add a **broadcast of the upper block’s last row** to vertically stitch row chunks in **H**.
- Write out; repeat for all `column_block_i` and `row_chunk_i`. ✅

---

## 1) What the kernel computes (the math)

For an input tensor `X[B=1, W, H, C]`, the **integral image** `I` is defined as:

$I(1, x, y, c) = \sum_{u=0}^{x}\sum_{v=0}^{y} X(1, u, v, c)$

The integral image kernel computes `I` with **tile blocks** to utilize Tenstorrent’s tiled compute model and maximize memory locality by storing data in SRAM-backed circular buffers (CBs). The work is split into three stages:

Throughout, the kernels use **tiles** of size `tile_height × tile_width` (often 32×32) and processes **blocks** of up to `block_depth` tiles (default 32), where `block_depth` iterates along **Width (W)** inside a row of tiles.

- **Reader:** moves input tiles from DRAM to on-chip CBs in the correct order; initializes the per-block prefix-sum state.
- **Compute:** performs cumulative sums along **Width (W)** within a block (kernel’s “axis 2”), then along **Height (H)** within a tile (kernel’s “axis 3”), and **propagates partial sums across blocks** so results are globally correct.
- **Writer:** writes results back to DRAM and, when on later row chunks, **imports the block above** to continue the global vertical accumulation (H propagation), broadcasting its last row to all rows.
- **Prefix sums** (running sums)

A **prefix sum** computes the sum of all elements up to and including a given position — an **inclusive** prefix sum:

$S_i = \sum_{k=0}^{i} x_k$

In the integral image kernel, this concept is applied along the **Width (W)** dimension first, and then along **Height (H)**:

- For each block, the kernel performs a **cumsum along W** (“`cumsum2`”), producing the running sum of tiles within that block.
- The **last tile** of the block is saved as a **carry** — the accumulated sum of everything so far.
- The next block **applies that carry** to its own cumsum results before continuing the scan.
- This repeats for subsequent blocks, ensuring the sums are globally correct across the entire image.

Intuitively:
> “Compute the cumsum of block 0 → keep the carry → cumsum of block 1 → apply carry → keep new carry → repeat sequentially for all following blocks.”
---

## 2) Key vocabulary (TT-metal essentials)

- **Tile:** smallest unit moved/operated in compute (e.g., 32×32 scalars).
- **Block:** a chunk of up to `block_depth` tiles traversed along **W** in the inner loop.
- **CB (Circular Buffer):** SRAM-backed FIFO used for staging tiles used to synchronize and move data inside and between kernels on the device.
- **`tile_regs_*` / `pack_*`:** register acquisition / data-format and packing API for tile ops.
- **`add_tiles`, `copy_tile`, `cumsum_tile`:** building blocks to compose per-tile math.
- **`noc_async_*`:** DMA across the NoC between DRAM ↔ L1.
- **`get_tile_id(...)`:** computes the logical tile index for a given (row chunk in **H**, column block in **W**, channel slice, inner tile in block).

---

## 3) Global tiling layout

We split the 2D image area `W×H` into a grid of **row chunks** × **column blocks**:

```
row_chunk_i = 0 ... num_blocks_in_column-1   (covers Height H, top → bottom)
column_block_i = 0 ... num_blocks_in_row-1   (covers Width  W, left → right)
inner tile in block: tile_i = 0 ... block_depth-1   (advances along W inside the block)
```

- Each `(row_chunk_i, column_block_i)` addresses a **block** of up to `block_depth` tiles along **W**.
- **Channels (C)** are split across cores: each core handles `my_channel = core_y * cores_x + core_x`.

---

## 4) Reader kernel — orchestrating input & initializing state

### 4.1 Functions and intent

- `zero_buffer(write_addr, bytes)`
  Efficiently zeroes a region in L1 by copying from a pre-zeroed memory (`MEM_ZEROS_BASE`) via async NoC reads. Used to reset the “start” tile.

- `prepare_start_tile_for_cumsum_axis_2(cb_start, tile_elems)`
  Reserves one tile in `cb_start` and **fills it with zeros**. This tile is used as the initial accumulator when starting a **W-scan** (left→right) within a block.

- `send_block(..., cb_input, ..., block_depth)`
  Computes `read_tile_id` for each tile in the current block and **loads** them into `cb_input` in processing order (left→right along **W**).

### 4.2 Control flow

For `B=1` (**only supported batch size**):
1. For each **row chunk** (advances in **H**),
2. For each **column block** (advances in **W**):
   - Create a **zero “start” tile** in `cb_start`.
   - Compute `block_depth = min(remaining W, ctas.block_depth)`.
   - `send_block(...)` DMA loads the tiles of this block into `cb_input` in order.

This guarantees the compute stage sees tiles as a stream **from left→right** within a row chunk, with a known initial accumulator (all-zeros) for the first tile of a horizontal scan.

---

## 5) Compute kernel — turning tiles into an integral image

The compute stage is the heart of the algorithm. It does three things repeatedly to produce a *globally correct* integral image:

1. **Width (W) cumulative sum** within the current block (kernel’s “axis 2”).
2. **W propagation across blocks** so later blocks see the correct left-of-block prefix.
3. **Height (H) cumulative sum** within each tile (kernel’s “axis 3”), plus **H propagation** across row chunks.

### 5.1 W cumulative sum: `cumsum_cube_axis_2(...)`

**What it does:** Given the block’s input tiles in `cb_input`, computes a left→right prefix sum across the `block_depth` tiles (**W** direction). It uses:

- `cb_start` — the zero tile you prepared; used only for the **first** tile of the block.
- `cb_acc` — rolling accumulator that stores the running sum after each tile.
- `cb_cumsum_stage_0` — the within-block W-prefix result for each tile.
- Optionally `cb_axis_2_buffer` — if `save_last_tile=true`, it **saves the last tile** of the block’s W-prefix for later cross-block propagation.

For each tile in the block:
- Use `cb_op = cb_start` for tile 0, else `cb_acc` (previous sum).
- `add_tiles(cb_input, cb_op)` → `WORKING_REG` gives the W-prefix for this tile.
- Push to `cb_acc` and copy to `cb_cumsum_stage_0`.
- If it’s the last tile **and** `save_last_tile`, also write it to `cb_axis_2_buffer` (needed by the next block along **W**).

### 5.2 W propagation across blocks: `propagate_tile_into_cube(...)`

When `column_block_i > 0` (not the first block in the row), each tile’s prefix must be **offset** by the **total sum of all tiles to the left**. That left-of-block total is the **saved last tile** from the previous block in **W** (`cb_axis_2_buffer`).

For each tile:
- Read per-tile prefix from the current block (`cb_cumsum_stage_a`).
- Add the **broadcast tile** from `cb_axis_2_buffer`.
- Write to `cb_cumsum_stage_b`.
- On the **last tile**, it optionally:
  - Pops `cb_axis_2_buffer` (releasing it for the next row chunk), and
  - If `save_last_tile` is requested, **re‑saves the last tile** for the next block.

This yields **globally correct horizontal prefixes** for the current block.

### 5.3 H cumulative sum within tile: `cumsum_cube_axis_3(...)`

Applies a **vertical** per-tile `cumsum_tile()` along rows (top→bottom in **H**) inside each 2D tile. The result is pushed either to an intermediate (`cb_cumsum_output`) if we’ll still add the **upper block** contribution, or directly to `cb_output` if this is the **first row chunk** (topmost).

### 5.4 H propagation (add from the upper block): `get_and_propagate_adder_cube(...)`

When `rows_block_i > 0`, the integral value must include **all rows above**. The code:

- Reads tiles from `cb_axis_3_buffer_read` (prepared by the writer; see §6.2).
- Adds them to the current block’s vertical cumsum and writes to `cb_output`.

### 5.5 Putting it together: `perform_intimg_along_row_chunk(...)`

For each column block in the current row:

- Always do `cumsum_cube_axis_2(...)` (horizontal scan), saving last tile if there’s a **next** block.
- If `column_block_i > 0`: perform **axis‑2 propagation** using the last‑tile from the previous block.
- Then, if `rows_block_i > 0`, perform **axis‑3 cumsum** and **add the upper block**; else just **axis‑3 cumsum** to final output.

Pseudocode of the decision tree (greatly simplified):

```
if first_column_block:
    A = cumsum_axis2(block)                     # → stage0
    if first_row_chunk:
        output = cumsum_axis3(A)                # local vertical
    else:
        tmp = cumsum_axis3(A)
        output = tmp + axis3_buffer_from_upper
else:
    A = cumsum_axis2(block)                     # → stage0
    B = propagate_axis2(axis2_buffer, A)       # add left‑block prefix → stage1
    if first_row_chunk:
        output = cumsum_axis3(B)
    else:
        tmp = cumsum_axis3(B)                   # → stage2
        output = tmp + axis3_buffer_from_upper
```

---

## 6) Writer kernel — exporting results and feeding back vertical context

The writer both **writes `cb_output` tiles to DRAM** and, when moving to a new row chunk (i.e., `row_chunk_i > 0`), it **imports the block above** and turns its **last row** into a tile-sized broadcast so the compute stage can add it to the current block.

### 6.1 Basic export: `output_block(...)`

For every tile in the block, compute its `write_tile_id` (consistent with the reader’s addressing) and `write_to_dram(cb_output, ...)` — writing the final integral image tiles for the current `(row_chunk_i, column_block_i, channel)` region.

### 6.2 Import the upper block & broadcast last row (H propagation)

When `row_chunk_i > 0`:

- `receive_upper_block(...)` loads the **already written** output tiles of the **previous row chunk** (same `column_block_i`, same channel) into `cb_axis_3_buffer_0`.
- `broadcast_last_row_to_all_rows_in_cube(...)` extracts the **last row** of those tiles (bottom of the upper block along **H**) and **broadcasts** that row to all rows of a new tile, writing into `cb_axis_3_buffer_1`.

Intuition: The integral image value at `(y, x)` must include the sum of all rows above `y`. The last row of the **upper block’s** integral tile is precisely the cumulative sum **up to the last row of that block** for each column `x`. Broadcasting it over all rows gives a per‑tile matrix that can be **added uniformly** to the current block’s local vertical prefix.

This `cb_axis_3_buffer_1` is then consumed by the compute stage via `get_and_propagate_adder_cube(...)` to produce the final vertically consistent result.

---

## 7) Axis mapping cheat-sheet (coherent with **[B, W, H, C]**)

| Concept here            | This doc’s axis | Kernel function name     | Direction             |
|------------------------|-----------------|--------------------------|-----------------------|
| Batches                | **B** (fixed 1) | outer loop (assumed 1)   | n/a                   |
| **Width scan**         | **W**           | `cumsum_cube_axis_2`     | left → right          |
| **Width propagation**  | **W**           | `propagate_tile_into_cube` | left-edge → right block |
| **Height scan**        | **H**           | `cumsum_cube_axis_3`     | top → bottom          |
| **Height propagation** | **H**           | `get_and_propagate_adder_cube` | add “upper” block    |
| Channels               | **C**           | core sharding            | per-core slice        |

> The names “axis_2/axis_3” in the code are historical; for coherence with your requested layout we speak in **W/H** terms throughout.

---

## 8) Correctness sketch vs. classic formula (with **[1, W, H, C]**)

Let `P_W(y_block, x_block, i, c)` be the result of the **W-scan** for the `i`-th tile inside block `(row_chunk=y_block, column_block=x_block)` and channel `c`.
Let `L_W(y_block, x_block-1, c)` be the **last tile** of the **previous** block in the same row after the W-scan.

Then the **global** W prefix for tile `i` in block `x_block` is:

$H\_W(y\_block, x\_block, i, c) = P\_W(y\_block, x\_block, i, c) + \mathbf{1}\_{x\_block>0}\,L\_W(y\_block, x\_block-1, c)$.

Next, vertical accumulation within tile plus the last-row broadcast of the **upper** block `(y_block-1, x_block)` yields for each `(x, y)` inside the tile:

$I(1, x, y, c) = cumsum_H(H_W(·)) + 1_{y_block > 0} · broadcastLastRow(I(1, ·, endOfUpperBlock, c))$

This is the classic 2D scan split into **tile-local** cumsums plus **cross-block** additive propagations in **W** and **H**.

---

## 9) Performance & robustness notes

- **Streaming friendly:** Reader streams DRAM→L1 in compute order; writer streams results L1→DRAM.
- **Overlap & guards:** `ReadCBGuard` / `WriteCBGuard` + `cb_wait_front` / `cb_reserve_back` control back-pressure while allowing DMA/compute overlap.
- **Zero tile reuse:** `zero_buffer` reuses a pre-zeroed NoC region, avoiding explicit per-element stores.
- **Edge blocks:** `block_depth` is computed per block (`min(remaining, ctas.block_depth)`), so borders are naturally handled.
- **Numerics:** Integral images can grow large; ensure `input_number_type` / `output_number_type` (and format in `pack_reconfig_data_format`) have sufficient dynamic range.
- **Channels & cores:** Each core handles a slice of **C**; `get_tile_id(...)` keeps channel/row/column consistent across reader/writer/compute.
- **Batches:** Current code assumes `B=1`; multi-batch would require disciplined buffer reuse across batches.

---

## 10) Walk‑through on a tiny example

Imagine `tile = 4×4`, `block_depth=2`, and a row has 3 blocks `(B0, B1, B2)`:

1. **Reader** zeros `cb_start`, streams `B0` tiles to `cb_input`.
2. **Compute** does horizontal cumsum on `B0` → `cb_cumsum_stage_0`, saves last tile to `cb_axis_2_buffer`, does vertical cumsum → `cb_output` (first row chunk: no upper add).
3. **Writer** writes `B0` to DRAM.
4. **Reader** streams `B1`. **Compute** uses `cb_axis_2_buffer` from `B0` to offset `B1`’s horizontal prefixes; then vertical cumsum; write to output. Update `cb_axis_2_buffer` with `B1`’s last tile for `B2`.
5. Next row chunk: **Writer** first loads the **upper** output block from DRAM and broadcasts its last row to `cb_axis_3_buffer_1` so Compute can add it during vertical propagation.

---

## 11) Signals & buffers (by role)

```
cb_start              # one zero tile for the first add in horizontal cumsum
cb_input              # reader → compute stream of input tiles
cb_acc                # rolling accumulator for cumsum along axis 2
cb_cumsum_stage_0     # horizontal cumsum result (pre-propagation)
cb_cumsum_stage_1     # after axis-2 propagation (if needed)
cb_cumsum_stage_2     # vertical cumsum intermediate (if upper add is needed)
cb_axis_2_buffer      # holds the last tile of previous block (horizontal propagation)
cb_axis_3_buffer_0    # writer’s import of upper block’s output (raw)
cb_axis_3_buffer_1    # broadcast(last_row(upper)) to add into current block
cb_output             # final per-tile output to write
```

---

## 12) Diagrams

### a) 📐 What’s an Integral Image (toy 4×4)
A tiny example to show “sum above-left, inclusive”.
```
Input I (4×4):          Integral S (4×4):
[1 2 0 3]               [1  3  3  6]
[0 1 4 0]   ──► S(y,x)  [1  4  8  11]
[2 0 2 1]   = Σi≤y Σj≤x [3  6  12 16]
[0 3 1 2]               [3  9  16 21]
```

### b) ↕️↔️ Two-Pass View (cumsum over height then width)
Shows that integral = cumsum(cumsum(I, height), width).
```
Step A: vertical cumsum (per column)
I:                   A = cumsum(I, height)
[1 2 0 3]           [1  2  0  3]
[0 1 4 0]   --->    [1  3  4  3]
[2 0 2 1]           [3  3  6  4]
[0 3 1 2]           [3  6  7  6]

Step B: horizontal cumsum (per row)
S = cumsum(A, width)
[1  2  0  3]   ->   [1  3  3  6]
[1  3  4  3]   ->   [1  4  8  11]
[3  3  6  4]   ->   [3  6  12 16]
[3  6  7  6]   ->   [3  9  16 21]
```

### c) ➕ Horizontal Carry (axis_2_buffer)
How we stitch across blocks in a row: save last tile’s prefix and add it to the next block.
```
Block k (tiles t0..tN):      Block k+1 (tiles u0..uM):
[ t0 t1 ... tN ]             [ u0 u1 ... uM ]
          │                         │
          └─ save last tile ───────►│  axis_2_buffer
                                    │
Add axis_2_buffer to every ui in block k+1 to preserve cumulative sum continuity.
```

### d) ⬇️ Vertical Carry via Last-Row Broadcast
How the writer turns the block above into a per-row baseline for the next row-chunk.
```
Integral tile from row-chunk r-1 (above):
[ a0
  a1
  ...
  a31 ]  (per column values)

Take LAST ROW (index 31)  →  [ v0 v1 ... v31 ]

Broadcast down:
[ v0 v1 ... v31 ]  (row 0)
[ v0 v1 ... v31 ]  (row 1)
[ v0 v1 ... v31 ]
   ...
[ v0 v1 ... v31 ]  (row 31)

This broadcasted tile is added to every tile in row-chunk r (vertical continuity).
```

### e) 🕒 Per-Block Timeline (one row-chunk)
A step-by-step of what happens for each block along the row.
```
[Reader]
  prepare zero → load tiles
      │
      ▼
[Compute]
  1) horizontal cumsum (save last tile if more blocks)
  2) if not first block: add axis_2_buffer
  3) vertical cumsum (per tile)
  4) if not first row-chunk: add axis_3 vertical carry
      │
      ▼
[Writer]
  write tiles to DRAM
  if next row-chunk exists: read tiles above → broadcast last row → publish axis_3 carry
```

### f) A quick ASCII W/H reference
```
       (row chunk along H: y = top → bottom)
  B0     B1      B2     ...       # blocks along W (left → right)

  ┌─► W-scan (cumsum_axis_2) ─┐
  │                           │ save last tile → axis2_buffer
  │                           ▼
  │                W propagation (add left block)
  │                           │
  │                           ▼
  │                H-scan (cumsum_axis_3) inside tile
  │                           │
  │        + broadcast(last_row from upper block)  (if y>0)
  │                           │
  └───────────────────────────▼
                          write DRAM
```


---

## Appendix: Code-to-concept map

- `cumsum_cube_axis_2` → horizontal prefix, plus optional “last tile” capture.
- `propagate_tile_into_cube` → add left‑of‑block cumulative to current block.
- `cumsum_cube_axis_3` → tile‑local vertical cumsum.
- `get_and_propagate_adder_cube` → add upper‑block broadcast (vertical stitch).
- Reader `send_block`/Writer `output_block` mirror each other’s `get_tile_id` addressing.
- Writer’s `broadcast_last_row_to_all_rows_in_cube` prepares the vertical offset matrix.
