# 03 — Tiles and numbers

*What the data looks like in memory, and what precision you're working in.*

---

## The tile

Tensix hardware does not operate on individual numbers, or on rows, or on
arbitrary slices. Its native unit is a **32 × 32 block of numbers**, called a
**tile**.

```
        32 columns
      ┌───────────────┐
  32  │               │
 rows │   one tile    │   = 1024 numbers
      │               │
      └───────────────┘
```

In `bfloat16` (2 bytes per number) a tile is `32 × 32 × 2` = **2048 bytes**.

Everything is expressed in tiles: buffers hold tiles, queues pass tiles, the
math engine multiplies tiles. Almost every kernel you write is a loop over tile
indices. If you find yourself thinking about individual numbers, you've usually
taken a wrong turn.

### Why 32×32?

Because the matrix engine is built to multiply two 32×32 tiles as a single
operation. Fixing the size lets the hardware hard-wire the data paths. Sizes
that aren't multiples of 32 get padded — a 40-row tensor occupies 2 tile-rows,
with 24 rows of padding you don't see.

---

## Tile layout

A tensor stored in **`TILE_LAYOUT`** is a sequence of tiles, laid out
**row-major over the tiles**.

Take a tensor of shape `[1, 1, 64, 128]`. In tiles that's 2 rows × 4 columns:

```
 tile grid (2 x 4)              page index in memory

┌──────┬──────┬──────┬──────┐   ┌───┬───┬───┬───┐
│ (0,0)│ (0,1)│ (0,2)│ (0,3)│   │ 0 │ 1 │ 2 │ 3 │
├──────┼──────┼──────┼──────┤   ├───┼───┼───┼───┤
│ (1,0)│ (1,1)│ (1,2)│ (1,3)│   │ 4 │ 5 │ 6 │ 7 │
└──────┴──────┴──────┴──────┘   └───┴───┴───┴───┘
```

**The formula you will use constantly:** for a matrix that is `Ct` tiles wide,

```
tile (r, c)  →  page index  r * Ct + c
```

Each page is one tile, 2048 bytes, contiguous.

So for a matmul `C = A @ B` where A is `Mt × Kt` tiles and B is `Kt × Nt`:

- `A[m][k]` is page `m * Kt + k`
- `B[k][n]` is page `k * Nt + n`
- `C[m][n]` is page `m * Nt + n`

Getting one of these backwards is the most common matmul bug, and it produces a
plausible-looking wrong matrix rather than an error.

### Inside a tile

A tile is internally divided into four 16×16 **faces**. You can ignore this
until you write SFPU code that indexes within a tile; it's why some APIs have a
`num_faces` argument. Nothing in this course needs it.

### Row-major vs tile layout

Ordinary "row-major" storage — the layout `torch` uses by default — puts all of
row 0 contiguously, then all of row 1. Tile layout does not. Converting between
them is a real operation (`tilize` / `untilize`), and the dojo's host code does
it for you when uploading and downloading tensors.

---

## Number formats

### A 30-second refresher on floating point

A floating-point number is stored as three parts:

```
  sign │ exponent │ mantissa
   1   │    8     │    23      = 32 bits  (float32)
```

- The **exponent** sets the scale — how big or small the number is.
- The **mantissa** sets the precision — how many significant digits you get.

More exponent bits → wider range. More mantissa bits → more accuracy.

### bfloat16

The format this course uses throughout:

```
  sign │ exponent │ mantissa
   1   │    8     │     7      = 16 bits  (bfloat16)
```

`bfloat16` keeps float32's **8 exponent bits** and throws away mantissa bits.
That's a deliberate choice: it has exactly the same *range* as float32 (so
nothing overflows when you convert), but only about **3 significant decimal
digits** of precision.

For neural networks this is a good trade — they turn out to be tolerant of noisy
values but intolerant of overflow. For your purposes here, it explains why the
grader never checks for exact equality.

> **Do not confuse it with `float16`**, which splits its 16 bits differently
> (5 exponent, 10 mantissa): more precision, far less range, and prone to
> overflowing.

### The formats you'll meet

| Format | Bytes per tile | Mantissa bits | Notes |
|---|---|---|---|
| `float32` | 4096 | 23 | Halves the DST registers available |
| `bfloat16` | 2048 | 7 | The default here |
| `bfp8_b` | 1088 | 7 (shared exponent) | Block float — see below |

**`bfp8_b`** is a Tenstorrent block-float format: a group of 16 values shares
one exponent, and each keeps its own small mantissa. That gets a tile down to
1088 bytes — nearly half of bfloat16.

Since most kernels are memory-bound (chapter 01), halving the bytes can nearly
halve the runtime. This is how production matmuls get their headline numbers.
It's out of scope for this course, but it's the first thing to reach for after
you finish it.

---

## What this means for checking correctness

With 7 mantissa bits, `bfloat16` can represent about 3 decimal digits. So:

- `0.1 + 0.2` will not be exactly `0.3`.
- Summing 64 numbers accumulates rounding error at every step.
- The same computation in a different *order* gives a slightly different answer
   — and the hardware's order is not torch's order.

The dojo therefore grades with two measures:

**PCC (Pearson correlation coefficient)** — how well the shape of your output
matches the reference, from -1 to 1. A value of `0.9999` means the outputs track
each other almost perfectly; `0.01` means your output is unrelated noise. This
is the standard accuracy metric across tt-metal, because it catches "the
algorithm is wrong" while tolerating "the last bit differs".

**Element-wise tolerance** — no single element may differ by more than an
absolute/relative bound. This catches the case where one value is wildly wrong
but the overall correlation stays high.

Both must pass. The one exception is lesson 01, a pure copy with no arithmetic:
that is checked for exact bit equality, because there's nothing to round.

A useful diagnostic habit: if PCC is near 1 but a few elements fail tolerance,
you probably have an indexing bug affecting a small region. If PCC is near 0,
the whole computation is wrong.

---

**Next:** [04 — Circular buffers](04-circular-buffers.md) — how kernels hand
tiles to each other.
