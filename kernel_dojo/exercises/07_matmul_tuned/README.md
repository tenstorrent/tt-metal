# 07 — Matmul at scale: reuse, parallelism, fidelity

**Goal:** take the lesson-06 matmul from "correct" to "fast". Three
optimisations, each measurable independently: **operand reuse**, **multi-core**,
and **math fidelity**.

This is the capstone. It's also the lesson where the benchmark matters more than
the diff.

> **Background:** [`theory 08 — Performance`](../../theory/08-performance.md) for
> how to read the benchmark, and
> [`theory 01 §7`](../../theory/01-latency-and-throughput.md) for what
> *memory-bound* and *arithmetic intensity* mean. This lesson's benchmark is
> mostly an exercise in interpreting numbers.

---

## Theory

### The lesson-06 kernel wastes almost all of its DRAM traffic

Count the reads. For every one of the `Mt × Nt` output tiles, it streams `Kt`
tiles of A and `Kt` tiles of B:

```
tiles read = 2 × Mt × Nt × Kt
```

But the matrices themselves are only `Mt×Kt + Kt×Nt` tiles. At `Mt=Nt=64,
Kt=8` that's **65536 tiles read to touch 1024 tiles of data — 64× redundant.**

DRAM traffic like that guarantees you never see the FPU's peak, no matter how
many cores you throw at it. The fix is **reuse**: get a piece of data into L1
once, and use it for as much output as you can before dropping it.

### Reuse by row

Reorganise so each core owns whole **rows** of C. Within a row `m`:

```
C[m][0] = Σ A[m][k] @ B[k][0]
C[m][1] = Σ A[m][k] @ B[k][1]
...
```

Every one of those uses the *same* `Kt` tiles of A. So load A's row once, keep
it resident in a CB, and stream only B past it:

```
tiles read = Mt × Kt  +  Mt × Nt × Kt
```

A's traffic drops by a factor of `Nt`. That's the single biggest change in this
exercise.

The kernel-side consequence: **don't pop `cb_a`** until the whole row is done.

```cpp
cb_wait_front(cb_a, Kt);              // A's row, resident for the whole row
for (uint32_t nt = 0; nt < Nt; nt++) {
    cb_wait_front(cb_b, Kt);
    tile_regs_acquire();
    for (uint32_t kt = 0; kt < Kt; kt++) {
        matmul_tiles(cb_a, cb_b, kt, kt, 0);   // index *into* the window
    }
    tile_regs_commit();
    ...pack...
    cb_pop_front(cb_b, Kt);           // B is consumed
}
cb_pop_front(cb_a, Kt);               // A only now
```

Notice `matmul_tiles(cb_a, cb_b, kt, kt, 0)` — with `Kt` tiles visible in the
window, the CB-relative index is `kt`, not `0`. This is the same windowing idea
as lesson 05, used for reuse rather than batching.

### Parallelism by row

Splitting by rows of C makes the multi-core version trivial: give each core a
contiguous range of `m`. No core needs another core's data, and B is read
independently (and redundantly) by each — which is fine, because B is small and
DRAM reads of the same address across cores hit the same banks efficiently.

The catch is **granularity**: with `Mt` rows and `C` cores you can't use more
than `Mt` cores, and the split quantises. `Mt = 64` over 64 cores is perfect;
`Mt = 64` over 48 cores means some cores do 2 rows and some do 1, so you pay for
2 while using 1.5.

### Math fidelity is a real speed knob

Set on the host in `ComputeConfigDescriptor`:

| Mode | Passes | Speed | Notes |
|------|--------|-------|-------|
| `LoFi`  | 1 | 4× | ~5 mantissa bits. Fine for bfp8, lossy for bfloat16 |
| `HiFi2` | 2 | 2× | Captures all 8 bits of a bfloat16 mantissa |
| `HiFi3` | 3 | 1.33× | |
| `HiFi4` | 4 | 1× | Full precision. Lesson 06 used this |

The FPU multiplies mantissas in slices; each extra pass covers another slice.
For **bfloat16 inputs, `HiFi2` is normally the right answer** — it retains
everything the inputs actually carry, at half the cost of `HiFi4`. `LoFi`
genuinely throws away input precision, and the benchmark reports PCC so you can
see how much.

It is a one-line change that can be worth 4×, and it is also the one most likely
to quietly cost you accuracy — so measure both. But *whether* it is worth
anything depends entirely on whether the FPU is what you're waiting on. The
benchmark in this lesson will make that point rather sharply.

---

## Your task

Write the reader and compute kernels for a row-parallel, A-reusing matmul.

Each core is assigned `n_rows` consecutive tile-rows of C starting at
`start_row`.

- **`kernels/reader.cpp`** — for each of this core's rows `m`: push A's row
  (`Kt` tiles) into CB 0 **once**, then for each `nt` push B's column (`Kt`
  tiles) into CB 1.
- **`kernels/compute.cpp`** — for each row: wait for `Kt` A tiles, then for each
  of `Nt` output tiles run the `Kt`-deep accumulation against B, packing one
  tile each time. Pop B per column; pop A once per row.
- **`kernels/writer.cpp`** — provided.

### What the host gives you

**`reader.cpp`**

| | |
|---|---|
| compile-time args | CB `a`, CB `b`, then accessor args for `a`, then `b` |
| runtime arg 0 | `a` base address |
| runtime arg 1 | `b` base address |
| runtime arg 2 | `Kt` |
| runtime arg 3 | `Nt` |
| runtime arg 4 | `start_row` — this core's first tile-row of C |
| runtime arg 5 | `n_rows` — how many rows this core owns |

**`compute.cpp`**

| | |
|---|---|
| compile-time args | CB `a`, CB `b`, CB out |
| runtime args | `Kt`, `Nt`, `n_rows` |

> The compute kernel doesn't need `start_row` — it only counts work.

**`writer.cpp`** gets `out` address, `start_row * Nt` as a first tile index, and
`n_rows * Nt` as a count.

### Run it

```bash
./dojo test 07
./dojo bench 07
```

The benchmark runs two sweeps: core scaling at fixed fidelity, and fidelity at
both 1 core and 64 cores.

---

## Hints

<details>
<summary>Reader structure</summary>

```cpp
const uint32_t end_row = start_row + n_rows;
for (uint32_t mt = start_row; mt < end_row; mt++) {
    // A's row, once
    cb_reserve_back(cb_a, Kt);
    uint32_t base_a = get_write_ptr(cb_a);
    for (uint32_t kt = 0; kt < Kt; kt++) {
        noc_async_read_page(mt * Kt + kt, a, base_a + kt * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_a, Kt);

    // B's columns, streamed
    for (uint32_t nt = 0; nt < Nt; nt++) {
        cb_reserve_back(cb_b, Kt);
        uint32_t base_b = get_write_ptr(cb_b);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            noc_async_read_page(kt * Nt + nt, b, base_b + kt * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_b, Kt);
    }
}
```

</details>

<details>
<summary>First row is right, later rows are wrong</summary>

You're popping `cb_a` in the wrong place. It must be popped exactly once per
row, *after* the `nt` loop — not inside it, and not never (which hangs once the
CB fills).

</details>

<details>
<summary>Everything is wrong by a consistent factor</summary>

Check the CB-relative indices in `matmul_tiles`. With `Kt` tiles visible, you
want `matmul_tiles(cb_a, cb_b, kt, kt, 0)`. Passing `0, 0` multiplies the first
tile of each window `Kt` times.

</details>

---

## Reading the benchmark

Measured on a Wormhole n150 at `Mt=64, Kt=8, Nt=32`:

| config | time | bandwidth | throughput |
|--------|------|-----------|------------|
| 1 core, HiFi4 | 2025 µs | 19 GB/s | 0.53 TFLOP/s |
| 8 cores, HiFi4 | 275 µs | 141 GB/s | 3.9 TFLOP/s |
| 32 cores, HiFi4 | 201 µs | 193 GB/s | 5.3 TFLOP/s |
| 64 cores, HiFi4 | 207 µs | 187 GB/s | 5.2 TFLOP/s |
| 64 cores, HiFi2 | 209 µs | 186 GB/s | 5.1 TFLOP/s |
| 64 cores, LoFi | 208 µs | 186 GB/s | 5.2 TFLOP/s |

**Reuse.** Compare against lesson 06 at `8×8×8`: reuse is why this kernel does
a 4× larger problem in a comparable time. The traffic line in the output tells
the story — it counts `Mt×Kt + Mt×Nt×Kt` tiles instead of `2×Mt×Nt×Kt`.

**Core scaling.** 1 → 8 cores is nearly 7.4×. 8 → 32 is only 1.4×, and 64 cores
is *slower* than 32. Bandwidth pins at ~190 GB/s from 32 cores on — the same
ceiling lesson 04 found. Past the knee, extra cores just contend.

**Fidelity — the interesting one.** `LoFi`, `HiFi2` and `HiFi4` are all within
1% of each other, at 1 core and at 64. A 4× reduction in math work bought
nothing.

That is not a broken measurement, it's the answer to a question you should
always ask before optimising: **is the FPU what I'm waiting on?** Here it isn't.
At 64 cores the kernel sits on the DRAM ceiling; at 1 core it's limited by that
core's own data movement. The multiplier is idle either way, so making it faster
changes nothing.

Optimising the part that isn't the bottleneck is the most common way to waste an
afternoon. Lesson 08 raises the arithmetic intensity until this kernel finally
stops being memory-bound — and then looks again at what's limiting it.

---

## Going further

The next steps beyond this course, roughly in order of payoff:

- **Block the output.** Compute a `2×4` block of C tiles per DST acquire (DST
  holds 8), so B tiles are reused across the block too.
- **Multicast.** Cores in the same row of the grid all need the same B tiles.
  `noc_async_write_multicast` lets one core fetch and broadcast to the rest,
  cutting B's DRAM traffic by the grid width. This is what
  `matmul_multicore_reuse_mcast` in `tt_metal/programming_examples/` does.
- **bfp8_b.** Tenstorrent's 8-bit block-float format halves the bytes per tile
  and lets `LoFi` be accurate, which is how production matmuls get their
  headline numbers.
