# The 2D pre-all-gather RMSNorm op: what it does, why it hung, and the fix

## 1. What the op computes

RMSNorm divides each row of a tensor by the square root of the mean of the squares of that row's values. To do that, it must first know, for every row, the sum of the squares of its values: sum(x²).

In a large model the hidden dimension (the last dimension, the row width) is often split across several devices. No single device then holds a whole row, so the sum is done in two steps:

1. Each device computes sum(x²) over its own part of every row. This is `ttnn.rms_norm_pre_all_gather`.
2. A separate op, `ttnn.all_gather`, copies every device's partial sums to every device. A second norm op, `ttnn.rms_norm_post_all_gather`, then combines the partial sums into the mean of x² for each row and normalizes the rows.

This document is about step 1 only. The code for this op is shared with LayerNorm and lives under `layernorm_distributed/`, so the file names below say "layernorm".

The input is a tensor of shape `(N, C, H, W)`. The op treats it as `N * C * H` rows of `W` values each. The output has one value per row: sum(x²) over the `W` values in that row.

## 2. Tiles

The hardware works on tiles of 32 × 32 values. A tensor in tile layout is stored as a grid of tiles. Two derived terms matter here:

- A **tile row** is 32 consecutive rows of the tensor, one tile high. The number of tile rows is `N * C * H / 32`.
- `Wt` is the row width in tiles: `W / 32`.

Tiles are numbered row by row: the tile in tile row `r` and tile column `c` has index `r * Wt + c`.

For an input of shape `(1, 1, 1024, 768)` there are 1024 / 32 = 32 tile rows, and `Wt` = 768 / 32 = 24.

The output is one tile wide. Column 0 of each output tile holds sum(x²) for each of the 32 rows in that tile row. So the output has one tile per tile row, and output tile `r` belongs to tile row `r`.

## 3. How the work is divided over cores

A Tenstorrent chip has a grid of cores. Each core has a position `(x, y)`, where `x` is the grid column and `y` is the grid row. On the Wormhole n150 used here, the usable grid is 8 columns by 8 rows, so 64 cores.

Each core computes sum(x²) for some of the rows. The **program factory** decides which core gets which tiles. It is code that runs on the CPU (the host), and it also builds the program that the cores run. For RMSNorm this op has two program factories, which divide the work in two different ways. (LayerNorm has a third, which this document does not cover.)

### 3.1 The 1D version: whole tile rows per core

By default, each core gets some whole tile rows. The code calls this the 1D version: its kernel and buffer names start with `PRE1D_` ([layernorm_pre_all_gather_program_factory.cpp:28-42](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L28-L42)), and a comment calls it the "normal (non-Welford, non-2D)" factory. The factory divides the tile rows as evenly as it can over up to 64 cores ([layernorm_pre_all_gather_program_factory.cpp:196-201](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L196-L201)). Each core then reads all `Wt` tiles of each of its tile rows, computes the full sum(x²) for those rows, and writes the result tiles itself ([layernorm_pre_all_gather_program_factory.cpp:363-388](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L363-L388)). No core needs data from another core.

Examples, all with `W = 768`, so `Wt = 24`:

- `(1, 1, 4096, 768)`: 128 tile rows. All 64 cores are used, with 2 tile rows each. Each core reads 2 × 24 = 48 tiles.
- `(1, 1, 1024, 768)`: 32 tile rows. 32 cores are used, with 1 tile row each. Each core reads 24 tiles. The other 32 cores do nothing.
- `(1, 1, 32, 768)`: 1 tile row. Only 1 core is used, and it reads all 24 tiles. The other 63 cores do nothing.

When the number of tile rows does not divide evenly, some cores get one tile row more than others. For example, 85 tile rows give 21 cores 2 tile rows each and 43 cores 1 tile row each.

So when the input has few tile rows, most cores do nothing, and the cores that do work each read a whole row width.

### 3.2 The 2D version: tile rows and row width both split

The 2D version is selected with `use_2d_core_grid=True` ([layernorm_pre_all_gather_device_operation.cpp:18-19](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_device_operation.cpp#L18-L19)). It divides the tile rows over the grid columns, and it also divides each tile row's width over the grid rows. So several cores share one tile row, each summing a different part of its width. These partial sums must then be added together, which the 1D version (section 3.1) never needs.

A small example: an input of shape `(1, 1, 64, 256)` has 2 tile rows and `Wt = 8`, so 16 tiles. The table shows which core `(x, y)` handles each tile:

| | tile column 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **tile row 0** | (0, 0) | (0, 1) | (0, 2) | (0, 3) | (0, 4) | (0, 5) | (0, 6) | (0, 7) |
| **tile row 1** | (1, 0) | (1, 1) | (1, 2) | (1, 3) | (1, 4) | (1, 5) | (1, 6) | (1, 7) |

- Tile row 0 goes to grid column `x = 0`, and tile row 1 to grid column `x = 1`.
- Within a tile row, tile column `c` goes to grid row `y = c`. So the 8 tiles of one tile row are spread down the 8 rows of one grid column, one tile per core.
- 16 cores are used, each reading 1 tile. Each core computes sum(x²) over its tile's 32 columns, which is only part of each row's sum. The 8 partial sums in grid column 0 must be added to get the full sums for tile row 0, and the same for grid column 1 and tile row 1.

The 1D version (section 3.1) would use 2 cores for this input, each reading all 8 tiles of one tile row.

In that example, `Wt` equals the number of grid rows, so each core gets exactly one tile. Usually it does not. Take `(1, 1, 64, 384)`: 2 tile rows and `Wt = 12`, so 24 tiles. 12 tiles cannot be split evenly over 8 grid rows, nor over 7. The factory uses the largest number of grid rows, at most 8, that divides 12 evenly: 6 grid rows, with 2 tiles per core:

| | tile column 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **tile row 0** | (0, 0) | (0, 0) | (0, 1) | (0, 1) | (0, 2) | (0, 2) | (0, 3) | (0, 3) | (0, 4) | (0, 4) | (0, 5) | (0, 5) |
| **tile row 1** | (1, 0) | (1, 0) | (1, 1) | (1, 1) | (1, 2) | (1, 2) | (1, 3) | (1, 3) | (1, 4) | (1, 4) | (1, 5) | (1, 5) |

- Tile row 0 still goes to grid column `x = 0`, and tile row 1 to grid column `x = 1`.
- Within a tile row, tile columns `2y` and `2y + 1` go to grid row `y`. For example, core `(0, 3)` reads tile columns 6 and 7 of tile row 0.
- 12 cores are used, each reading 2 tiles. Grid rows 6 and 7 are not used. Each grid column has 6 partial sums to add for its tile row.

The 1D version (section 3.1) would again use 2 cores, each reading all 12 tiles of one tile row.

If `Wt` is a prime larger than 8, for example 13, the only divisor of 13 that is at most 8 is 1. So the row width is not split at all: one core in grid row 0 reads all 13 tiles of each of its tile rows, and the other grid rows are not used. With 2 tile rows that is 2 cores, the same as the 1D version (section 3.1).

The factory computes four numbers ([layernorm_pre_all_gather_program_factory.cpp:494-504](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L494-L504)):

- `cores_x`: the largest divisor of the number of tile rows that is at most 8.
- `tiles_per_core_x`: tile rows per core, `number of tile rows / cores_x`.
- `cores_y`: the largest divisor of `Wt` that is at most 8.
- `tiles_per_core_y`: tile columns per core, `Wt / cores_y`.

The code takes both limits from the number of grid rows on the device, including the limit on `cores_x`, which counts grid columns. On the 8 × 8 grid of the n150 the two counts are the same.

Core `(x, y)` handles tile rows `[x * tiles_per_core_x, (x + 1) * tiles_per_core_x)` and, in each of them, tile columns `[y * tiles_per_core_y, (y + 1) * tiles_per_core_y)`.

The cores with the same `x` are in one grid column, and the code calls them a **column**. They handle the same tile rows, each a different part of the width. Core `(x, 0)` is the column's **merge core** (the code's term). Each core in the column computes a partial sum(x²) over its part of a row, and the merge core adds the column's partial sums to get the full sum for the row.

Applied to the examples above:

- `(1, 1, 64, 256)`: 2 tile rows, so `cores_x = 2` and `tiles_per_core_x = 1`. `Wt = 8`, so `cores_y = 8` and `tiles_per_core_y = 1`. This matches the first table.
- `(1, 1, 64, 384)`: `cores_x = 2` and `tiles_per_core_x = 1`, as above. `Wt = 12`, so `cores_y = 6` and `tiles_per_core_y = 2`. This matches the second table.
- `Wt = 13`: `cores_y = 1` and `tiles_per_core_y = 13`.

Applied to the inputs from 3.1:

- `(1, 1, 32, 768)`: 1 tile row, so `cores_x = 1` and `tiles_per_core_x = 1`. `Wt = 24`, so `cores_y = 8` (the largest divisor of 24 that is at most 8) and `tiles_per_core_y = 3`. 8 cores are used. The 1D version (section 3.1) would use 1 core. Core `(0, y)` reads tile columns `3y` to `3y + 2` of the single tile row, and merge core `(0, 0)` adds the 8 partial sums.
- `(1, 1, 1024, 768)`: 32 tile rows, so `cores_x = 8` and `tiles_per_core_x = 4`. `cores_y = 8` and `tiles_per_core_y = 3`, as above. All 64 cores are used; the 1D version (section 3.1) uses 32. Core `(2, 5)` handles tile rows 8 to 11 and, in each of them, tile columns 15 to 17: 4 × 3 = 12 tiles. In the 1D version (section 3.1) each core would read 24 tiles.
- `(1, 1, 4096, 768)`: 128 tile rows, so `cores_x = 8` and `tiles_per_core_x = 16`. All 64 cores are used, as in the 1D version (section 3.1).

The important fact for this document: **in the 2D version, a core handles more than one tile row whenever the input has more than 8 tile rows**, that is, when `N * C * H > 256`. The rest of this document is about the 2D version only.

## 4. The pieces of the program on each core

Each core runs up to three small programs, called kernels, in parallel:

- The **reader** copies input tiles from DRAM into the core's local memory (SRAM). It also sends this core's partial sum to the merge core.
- The **compute** kernel squares the input tiles and sums them.
- The **writer** (merge cores only) copies the final result tiles from SRAM to DRAM.

The kernels pass tiles to each other through **dataflow buffers**: queues of tiles in SRAM. Dataflow buffer is the code's name; the older name is circular buffer.
- The producer side calls `reserve_back(n)` to wait for free space for `n` tiles, writes the tiles, then calls `push_back(n)`.
- The consumer side calls `wait_front(n)` to wait until `n` tiles are present, reads them, then calls `pop_front(n)` to free them.

A `wait_front` with no matching `push_back`, or a `reserve_back` with no matching `pop_front`, blocks forever. A device hang in this op is one of these two cases.

Cores signal each other with **semaphores**: counters in SRAM. A core can increment a counter on another core over the network-on-chip (NoC), which connects the cores. A core can also wait until its own counter reaches a value.

## 5. How the program is meant to work for each tile row

This section describes the intended steps for each tile row. Section 6 describes where the code did not do this.

1. The reader puts this core's `tiles_per_core_y` input tiles into the input buffer.
2. The compute kernel squares them, then adds all the squares along each row. This row-wise add is a "reduce".
   - The hardware reduce multiplies the sum by the values in a constant **scaler tile**. Here the scaler tile holds 1.0, so the sum is unchanged.
   - The reader makes this tile once, at kernel start, and pushes it into a one-tile buffer ([reader_layernorm_preallgather_2d.cpp:68-72](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/reader_layernorm_preallgather_2d.cpp#L68-L72)).
   - Every reduce does `wait_front(1)` on that buffer.
3. The compute kernel puts the result, one tile holding this core's partial sum, into the one-tile **partial buffer**.
4. The reader takes the partial tile and writes it over the NoC into the merge core's **gather buffer**. The gather buffer has one slot per core in the column, so it holds `cores_y` tiles ([layernorm_pre_all_gather_program_factory.cpp:553](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L553-L553)). Then the reader increments a semaphore, `reducer`, on the merge core. The merge core's reader does the same with its own partial, into its own gather buffer.
5. On the merge core, the reader waits until `reducer` reaches `cores_y`, which means every partial has arrived. It then calls `push_back(cores_y)` on the gather buffer, so that the compute kernel can read the partials, and resets `reducer` to 0.
6. The merge core's compute kernel adds the `cores_y` partials and pushes one result tile to the writer. The code that does this is called the merge block.
7. The writer writes that tile to the output tensor in DRAM.

## 6. The bug

The kernels were written for a core that handles one tile row. For example, with the input `(1, 1, 32, 768)` from section 3.2, every core handles only tile row 0, so each step of section 5 runs once on every core. The code ran each step once, which is correct for that input.

With more than one tile row per core, every step of section 5 must run once for each tile row, and each row's tiles are in a different place. Take the input `(1, 1, 1024, 768)` from section 3.2. It has 32 tile rows spread over `cores_x = 8` grid columns, so each core handles `tiles_per_core_x` = 32 / 8 = 4 tile rows. Its row width, `Wt = 24`, is spread over `cores_y = 8` grid rows, so each core handles `tiles_per_core_y` = 24 / 8 = 3 tile columns. By the rule in section 3.2, core `(2, 5)` handles tile rows 2 × 4 = 8 to 3 × 4 − 1 = 11, and tile columns 5 × 3 = 15 to 6 × 3 − 1 = 17. So:

- Core `(2, 5)`'s reader must read 3 tiles, tile columns 15 to 17, from each of its 4 tile rows. Tile rows 8 to 11 are shared by the 8 cores of grid column 2, 3 tile columns each. The table shows the tile numbers each of these cores reads, using the numbering from section 2 (`r * Wt + c`, here `r * 24 + c`). Tile rows 0 to 7 come first, with 24 tiles each, so tile row 8 starts at tile 8 × 24 = 192. Core `(2, 5)`'s tiles are in bold:

  | tile row | columns 0–2: core (2, 0) | columns 3–5: core (2, 1) | columns 6–8: core (2, 2) | columns 9–11: core (2, 3) | columns 12–14: core (2, 4) | columns 15–17: core (2, 5) | columns 18–20: core (2, 6) | columns 21–23: core (2, 7) |
  |---|---|---|---|---|---|---|---|---|
  | 8 | 192–194 | 195–197 | 198–200 | 201–203 | 204–206 | **207–209** | 210–212 | 213–215 |
  | 9 | 216–218 | 219–221 | 222–224 | 225–227 | 228–230 | **231–233** | 234–236 | 237–239 |
  | 10 | 240–242 | 243–245 | 246–248 | 249–251 | 252–254 | **255–257** | 258–260 | 261–263 |
  | 11 | 264–266 | 267–269 | 270–272 | 273–275 | 276–278 | **279–281** | 282–284 | 285–287 |

  Each row of the table starts 24 tiles (one full row width, `Wt`) after the previous one. So after tile 209, the next tile core `(2, 5)` needs is 231, not 210. Tiles 210 to 215 belong to cores `(2, 6)` and `(2, 7)` in tile row 8, and tiles 216 to 230 to cores `(2, 0)` to `(2, 4)` in tile row 9.
- Core `(2, 5)`'s compute kernel must run the reduce 4 times, each time with the same scaler tile, which the reader made only once.
- Core `(2, 5)`'s reader must send 4 partial sums to merge core `(2, 0)`, one per tile row. Merge core `(2, 0)` must then merge 4 times, and its writer must write 4 output tiles, tiles 8 to 11.

The code got all three wrong. Section 6.1 covers the scaler tile and section 6.2 the partial sums: both make the program hang. Section 6.3 covers the tile positions: the program also reads and writes the wrong tiles.

### 6.1 The scaler tile was removed after the first row

The compute kernel called `pop_front(1)` on the scaler buffer at the end of every row. The reader pushes the scaler tile only once. So after row 0 the buffer is empty, and the reduce for row 1 waits for a scaler tile that never comes. Every compute kernel on every core hangs there.

### 6.2 Partial sums were sent and merged only once

Suppose the scaler problem from 6.1 is fixed, so the compute kernel can run its reduce for every tile row. The program still hangs, because the code did steps 4 to 6 of section 5 only once per core, not once per tile row:

- The reader first read the input for all of its tile rows. Then it took one partial sum out of the partial buffer, sent it to the merge core, and finished.
- The merge block ran once, after all tile rows, so each merge core produced one result tile.

The partial buffer holds one tile (section 5, step 3). So before the compute kernel can put in the next partial sum, it calls `reserve_back(1)` and waits until the reader has taken the previous one out.

**With more than one tile row per core, the program hangs. Where it hangs depends on the number of tile rows.** The reader takes only the first partial sum out of the partial buffer, which frees room for one more. The second partial sum thus still fits into the buffer. However, the reader never takes the second partial sum out of the buffer, so a third partial sum does not fit and the compute kernel waits forever for space.

**3 or more tile rows: the compute kernel hangs.** Core `(2, 5)` from the example above has 4 tile rows, 8 to 11:

1. The compute kernel puts the partial sum for tile row 8 into the partial buffer. The buffer is now full.
2. The reader takes that partial sum out and sends it to merge core `(2, 0)`. The buffer is empty again. That was the reader's only send, so the reader finishes.
3. The compute kernel puts the partial sum for tile row 9 into the buffer. The buffer is full again, and no kernel will ever take this tile out.
4. For tile row 10, the compute kernel calls `reserve_back(1)` to wait for free space in the buffer. It waits forever.

**Exactly 2 tile rows: the compute kernel finishes, but the writer hangs.** For example, the input `(1, 1, 512, 768)` has 16 tile rows, so each core handles 16 / 8 = 2 of them:

1. The first partial sum is sent, and the second stays in the partial buffer, as in steps 1 to 3 just above. The compute kernel has no third tile row, so it never waits for space.
2. On each merge core, the merge block runs once and passes one result tile to the writer.
3. The writer was told to write `tiles_per_core_x` = 2 tiles, one per tile row ([layernorm_pre_all_gather_program_factory.cpp:793](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L793-L793)). It writes the first one and waits forever for the second.

### 6.3 Wrong tile positions (hidden by the hang)

- The reader's start tile was `x * Wt + y * tiles_per_core_y`. That is correct only with one row per core. Core `x` starts at tile row `x * tiles_per_core_x`, so with the numbering from section 2 the start must be `x * tiles_per_core_x * Wt + y * tiles_per_core_y`.
- Inside a core, the reader stepped from the last tile of its part of one row to the next tile index. That tile belongs to another core: the next core's part of the same row or, for the last core in the column, the first core's part of the next row. The step from the start of one row to the start of the next must be a full row width, `Wt` tiles.
- The writer's start tile was `x`. It must be `x * tiles_per_core_x`, the first tile row of core `x`.

With these errors, a version that did not hang would write sums of the wrong tiles to the wrong output positions, and report no error.

## 7. The fix

### 7.1 Scaler

The compute kernel now removes the scaler tile once, after the loop over tile rows ([layernorm_pre_allgather_2d.cpp:167-173](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_2d.cpp#L167-L173)). The merge block also uses a constant zero tile that the reader pushes once. Because the merge block now runs once per row (7.2), this tile is also removed only once, after the loop.

### 7.2 One send and one merge per row

The merge block now runs inside the loop over tile rows ([layernorm_pre_allgather_2d.cpp:115-164](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_2d.cpp#L115-L164)), so the merge core produces one result tile per row.

Steps 4 and 5 in the reader are now a function, `send_partial`, called once per row ([reader_layernorm_preallgather_2d.cpp:93-132](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/reader_layernorm_preallgather_2d.cpp#L93-L132)). The reader sends row `r`'s partial after it has read row `r + 1`'s input. So the compute kernel has input to work on while the reader waits for the merge core to accept the partial ([reader_layernorm_preallgather_2d.cpp:172-183](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/reader_layernorm_preallgather_2d.cpp#L172-L183)).

Sending once per row creates a new problem, because the gather buffer holds only one row's partials:
- A fast core could write its row 1 partial into the merge core's gather buffer before the merge core's compute kernel has read the row 0 partials. That write would overwrite one of the row 0 partials.
- A fast core could also increment `reducer` for row 1 just before the merge core resets it to 0 at the end of row 0. That increment would be lost.

To prevent both, a second semaphore, `gather_free`, is added ([layernorm_pre_all_gather_program_factory.cpp:815](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L815-L815)). For each row:

1. The merge core's reader calls `reserve_back(cores_y)` on its gather buffer. This waits until the compute kernel has popped the previous row's partials. The merge core already reset `reducer` to 0 at the end of the previous row, before this step.
2. The merge core then increments `gather_free` by 1 on every other core in its column, with one NoC write that reaches all of those cores at once (a multicast).
3. Each other core waits until its `gather_free` is 1, resets it to 0, and only then writes its partial and increments `reducer`.

So no core can write or count a partial for the next row until the merge core has finished with the current one. When `cores_y = 1`, the column has only the merge core, and the multicast and the wait are skipped.

### 7.3 Tile positions

- The factory computes the corrected start tiles ([layernorm_pre_all_gather_program_factory.cpp:758-759](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L758-L759)).
- The reader receives a new argument, `row_stride = Wt`, and starts each row `row_stride` tiles after the start of the previous one ([reader_layernorm_preallgather_2d.cpp:134-170](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/reader_layernorm_preallgather_2d.cpp#L134-L170)).

### 7.4 What did not change

The merge block contains a call to `compute_kernel_hw_startup`, which reconfigures the compute hardware.
- Its documentation says to call it only once, at kernel start. Issue #52395 tracks replacing such calls.
- The fix does not change this call. Because the merge block now runs once per row, the call now runs once per row on merge cores.
- The tests in section 8 pass with this, but the effect is not otherwise checked.
- Open pull request #57495 removes the call.

## 8. How it was checked

All checks ran on one Wormhole n150.

- **Hang reproduced without the fix.** The fix was removed and the code rebuilt.
  - The example input from issue #55075, `(1, 1, 1024, 1024)`, ended with `TIMEOUT: device timeout, potential hang detected`.
  - `(1, 1, 32, 64)`, with one row per core, passed.
  - `(1, 1, 288, 64)`, with 3 rows per core, timed out the same way.
- **Test extended.** The existing test `test_pre_all_gather_non_welford_fp32_precision` compares the op's sum(x²) with a float64 PyTorch result for float32 input. It now also runs the 2D version on four shapes with more than one row per core ([test_distributed_layernorm_pre_allgather.py:1065-1083](tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py#L1065-L1083)):

  | Shape | Rows per core | `cores_y` |
  |---|---|---|
  | (1, 1, 288, 64) | 3 | 2 |
  | (1, 1, 320, 256) | 2 | 8 |
  | (1, 1, 1024, 1024) | 4 | 8 |
  | (1, 1, 288, 32) | 3 | 1 |

  Like the existing cases, each shape runs with and without a residual input. A residual input is a second tensor that the op adds to the input before squaring. Each shape also runs with both the accurate and the fast summation mode of the hardware.
- **With the fix,** all 28 cases of that test pass: 12 that already existed and 16 new. Both test files for this op pass (166 tests): `test_distributed_layernorm_pre_allgather.py` and `test_distributed_rmsnorm_allgather.py`.
- **The test detects the bugs.**
  - With the reader's row step set to `tiles_per_core_y` instead of `Wt`, the three new shapes with `cores_y > 1` fail. With `cores_y = 1`, `tiles_per_core_y` equals `Wt`, so this change cannot cause a failure.
  - With every core's reader starting at tile 0, all 20 2D cases fail: the 16 new ones and the 4 existing ones that use the 2D version.

Not checked: other Tenstorrent chips (Blackhole, Quasar), and performance.
