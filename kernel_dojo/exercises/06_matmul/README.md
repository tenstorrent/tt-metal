# 06 — Matmul: the FPU's real job

**Goal:** `C = A @ B` for tile-aligned matrices. The first kernel with real
arithmetic in it, and the first where DST is used as an accumulator rather than
a mailbox.

> **Background:** [`theory 06 — Compute`](../../theory/06-compute.md), especially
> "Matmul, and its two traps". [`theory 03`](../../theory/03-tiles-and-numbers.md)
> has the page-index formula you'll need for indexing a 2-D matrix.

---

## Theory

### Matmul is tile-granular too

`matmul_tiles` multiplies one 32×32 tile by another and **accumulates** into a
DST slot:

```cpp
matmul_tiles(cb_a, cb_b, tile_a, tile_b, dst);   // DST[dst] += A_tile @ B_tile
```

That `+=` is the whole trick. Writing `C = A @ B` in tiles:

```
C[m][n] = Σ over k of  A[m][k] @ B[k][n]
```

is exactly a loop of `matmul_tiles` calls into the same DST slot. Nothing else
accumulates for you — the FPU does it inside DST, for free, at full rate.

`tile_regs_acquire()` **zeroes DST**, so the accumulator starts clean each time
you acquire. Acquire once per output tile, loop over K, then pack:

```cpp
tile_regs_acquire();                       // DST[0] = 0
for (uint32_t kt = 0; kt < Kt; kt++) {
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    matmul_tiles(cb_a, cb_b, 0, 0, 0);     // DST[0] += A @ B
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}
tile_regs_commit();
```

**Do not pack inside the K loop.** Packing to a bfloat16 CB and reading it back
would round the partial sum to 8 mantissa bits on every step. Keeping the
accumulation in DST keeps it at the FPU's internal precision, and it is the
reason a K=64 matmul isn't numerical mush.

### `SrcOrder::Reverse`

Matmul is the one operation where the operands map to the source registers
backwards: `in0` lands in **SrcB** and `in1` in **SrcA**. The hardware
configuration has to know that, so matmul's startup call differs from every
other op you've written:

```cpp
compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
matmul_init(cb_in0, cb_in1);
```

Use `SrcOrder::Regular` (the default) and you get silently wrong numbers, not an
error. This is a two-line detail that costs people an afternoon.

Note also that matmul does **not** use `binary_op_init_common` — it uses
`compute_kernel_hw_startup` plus `matmul_init`.

### Tile indexing in a 2-D matrix

A tile-layout matrix of `Mt × Kt` tiles stores tile `(m, k)` at linear page
index `m * Kt + k` — row-major over tiles. So:

- `A[m][k]` is page `m * Kt + k` of tensor A
- `B[k][n]` is page `k * Nt + n` of tensor B
- `C[m][n]` is page `m * Nt + n` of tensor C

The reader's job is to feed the compute kernel the K tiles of A's row `m` and
the K tiles of B's column `n`, in the same order, for each output tile.

### Math fidelity

The FPU trades accuracy for speed in four steps, set on the host side:

| Mode | Passes | Relative speed | What it keeps |
|------|--------|----------------|---------------|
| `LoFi`  | 1 | 4× | ~5 bits of mantissa |
| `HiFi2` | 2 | 2× | ~8 bits (all of bfloat16) |
| `HiFi3` | 3 | 1.33× | more |
| `HiFi4` | 4 | 1× | full bfloat16 × bfloat16 |

Each pass runs the multiplier over another slice of the mantissa. For bfloat16
inputs `HiFi2` is usually the sweet spot — it captures the entire input
mantissa, and `HiFi4` mostly buys precision the inputs never had.

This exercise runs `HiFi4`. Lesson 07 makes it a knob and you measure it.

### Where the time goes

Per output tile this kernel does `Kt` tile-matmuls — that's `Kt × 32³ × 2` ≈
`65536 × Kt` FLOPs — while reading `2 × Kt` tiles. That's roughly 16 FLOPs per
byte, an order of magnitude more than the element-wise add of lesson 03.

It is still nowhere near enough. This kernel re-reads *all* of B for every row
of A, so it moves `2 × Mt × Nt × Kt` tiles to touch `Mt×Kt + Kt×Nt` tiles of
data. Expect single-core numbers around 0.1 TFLOP/s — the FPU spends nearly all
its time waiting.

Lessons 07 and 08 are about closing that gap, and about learning to tell which
resource is actually holding you up.

---

## Your task

Write the reader and the compute kernel for a single-core matmul.

- **`kernels/reader.cpp`** — for each output tile `(mt, nt)`, feed CB 0 with the
  `Kt` tiles of A row `mt` and CB 1 with the `Kt` tiles of B column `nt`.
- **`kernels/compute.cpp`** — accumulate `Kt` tile-matmuls into one DST slot,
  then pack one output tile.
- **`kernels/writer.cpp`** — provided. Output tiles come out in linear order.

### What the host gives you

**`reader.cpp`**

| | |
|---|---|
| compile-time arg 0 | CB `a` (0) |
| compile-time arg 1 | CB `b` (1) |
| compile-time args 2.. | accessor args for `a`, then `b` |
| runtime arg 0 | `a` base address |
| runtime arg 1 | `b` base address |
| runtime arg 2 | `Mt` — tile rows of A |
| runtime arg 3 | `Kt` — inner dimension in tiles |
| runtime arg 4 | `Nt` — tile columns of B |

**`compute.cpp`**

| | |
|---|---|
| compile-time args | CB `a`, CB `b`, CB out |
| runtime args | `Mt`, `Kt`, `Nt` |

### API you need

```cpp
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"

compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);
matmul_init(cb_a, cb_b);
matmul_tiles(cb_a, cb_b, tile_a, tile_b, dst);   // accumulates
```

### Run it

```bash
./dojo test 06
./dojo bench 06
```

---

## Hints

<details>
<summary>Reader loop structure</summary>

```cpp
for (uint32_t mt = 0; mt < Mt; mt++) {
    for (uint32_t nt = 0; nt < Nt; nt++) {
        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_reserve_back(cb_a, 1);
            noc_async_read_page(mt * Kt + kt, a, get_write_ptr(cb_a));

            cb_reserve_back(cb_b, 1);
            noc_async_read_page(kt * Nt + nt, b, get_write_ptr(cb_b));

            noc_async_read_barrier();
            cb_push_back(cb_a, 1);
            cb_push_back(cb_b, 1);
        }
    }
}
```

The compute kernel's loop nest must match this order exactly — it consumes
whatever arrives, in arrival order.

</details>

<details>
<summary>Results are wrong but structured (not noise)</summary>

Check `SrcOrder::Reverse` first — with `Regular` you get a plausible-looking but
wrong matrix.

If that's right, you probably transposed an index: `k * Nt + n` for B, not
`n * Kt + k`. Try `Mt = Nt` with a `Kt` that differs, so a transposition
produces a shape error you can actually see.

</details>

<details>
<summary>Results drift as Kt grows</summary>

You're packing inside the K loop. Move `pack_tile` after the loop so the
accumulation stays in DST at full precision.

</details>

---

## Going further

- Set `Kt = 1` and confirm you get plain tile multiplication.
- Run `./dojo bench 06` and compare the two rates against their ceilings: ~195
  GB/s for DRAM (from lesson 04) and tens of TFLOP/s for the FPU. Neither is
  close here — a single core can't saturate either. Predict which one lesson
  07's optimisations will hit first, then check whether you were right.
