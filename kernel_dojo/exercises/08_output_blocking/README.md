# 08 — Output blocking: finding the real bottleneck

**Goal:** raise arithmetic intensity until the kernel stops being DRAM-bound —
then discover that the thing limiting it next is not what you'd guess.

This lesson is as much about *reading a benchmark* as about writing a kernel.

> **Background:** [`theory 08 — Performance`](../../theory/08-performance.md) and
> [`theory 01 §7`](../../theory/01-latency-and-throughput.md). The measured table
> at the bottom of this lesson is the clearest example in the course of a
> bottleneck moving, and it's worth having the vocabulary before you read it.

---

## Theory

### Lesson 07 still reads B far too many times

With A-row reuse, the traffic was:

```
A:  Mt × Kt              (each row once)
B:  Mt × Nt × Kt         (all of B, once per row of A)
```

B dominates completely. At `Mt=128, Kt=8, Nt=32` that's 1024 tiles of A and
**32768** of B.

The reason is that A's row is reused across `Nt` columns, but each B column is
used exactly once before being dropped. Reuse was one-dimensional.

### Block the output in two dimensions

Hold `Mb` rows of A resident instead of one. Now for each B column you compute
`Mb` output tiles:

```
        B column nt
             │
    ┌────────▼────────┐
A   │ row m0   ──────▶│ C[m0][nt]
row │ row m0+1 ──────▶│ C[m0+1][nt]     all from ONE read of B's column
blk │ row m0+2 ──────▶│ C[m0+2][nt]
    │ row m0+3 ──────▶│ C[m0+3][nt]
    └─────────────────┘
```

B's traffic drops by `Mb`:

```
A:  Mt × Kt
B:  (Mt / Mb) × Nt × Kt
```

The compute kernel keeps `Mb` independent accumulators, one DST slot each:

```cpp
tile_regs_acquire();
for (uint32_t m = 0; m < Mb; m++) {
    for (uint32_t kt = 0; kt < Kt; kt++) {
        matmul_tiles(cb_a, cb_b, m * Kt + kt, kt, m);
    }
}
tile_regs_commit();
```

Note the indices: A's tile `(m, kt)` sits at window slot `m * Kt + kt`, the B
tile is `kt` for every `m`, and the DST slot is `m`. DST caps `Mb` at 8.

### What you give up

`Mb` trades **parallelism for intensity**. There are `Mt / Mb` row-blocks, so
doubling `Mb` halves the number of independent work items. At `Mt=128, Mb=8`
there are only 16 blocks — you cannot use more than 16 cores. On a bigger
problem that's fine; on a small one it's a real cost.

L1 also grows: `cb_a` must hold `Mb × Kt` tiles.

---

## Your task

- **`kernels/reader.cpp`** — load an `Mb × Kt` sub-block of A per row-block,
  then stream B's columns as before.
- **`kernels/compute.cpp`** — `Mb` accumulators per B column.
- **`kernels/writer.cpp`** — provided. Note it *scatters*: the `Mb` tiles of a
  column are `Nt` apart in C, so it can't write a contiguous run.

Each core owns row-blocks `[start_block, start_block + n_blocks)`; row-block
`blk` covers rows `blk * Mb` upward.

### What the host gives you

**`reader.cpp`**

| | |
|---|---|
| compile-time args | CB `a`, CB `b`, `Mb`, then accessor args for `a`, `b` |
| runtime args | `a` addr, `b` addr, `Kt`, `Nt`, `start_block`, `n_blocks` |

**`compute.cpp`**

| | |
|---|---|
| compile-time args | CB `a`, CB `b`, CB out, `Mb` |
| runtime args | `Kt`, `Nt`, `n_blocks` |

### Run it

```bash
./dojo test 08
./dojo bench 08
```

---

## Hints

<details>
<summary>Reader: loading A's sub-block</summary>

```cpp
cb_reserve_back(cb_a, Mb * Kt);
const uint32_t base_a = get_write_ptr(cb_a);
for (uint32_t m = 0; m < Mb; m++) {
    for (uint32_t kt = 0; kt < Kt; kt++) {
        noc_async_read_page((row0 + m) * Kt + kt, a, base_a + (m * Kt + kt) * tile_bytes);
    }
}
noc_async_read_barrier();
cb_push_back(cb_a, Mb * Kt);
```

`Mb * Kt` transactions in flight at once — the batching idea from lesson 05,
now doing double duty as the reuse mechanism.

</details>

<details>
<summary>Output is right for m=0 and wrong for the rest</summary>

The A window index. It must be `m * Kt + kt`. If you wrote `kt`, every row of
the block multiplies A's first row.

</details>

<details>
<summary>Hangs at large Mb</summary>

`cb_a` needs at least `Mb * Kt` pages to satisfy a single `cb_reserve_back`.
The host allocates `2 * Mb * Kt`, so check you aren't reserving more than you
meant to.

</details>

---

## Reading the benchmark — the actual lesson

Measured on a Wormhole n150, `Mt=128, Kt=8, Nt=32`, 16 cores:

| `Mb` | time | traffic | bandwidth | throughput |
|------|------|---------|-----------|------------|
| 1 | 394 µs | 74 MiB | **197 GB/s** | 5.4 TFLOP/s |
| 2 | 222 µs | 42 MiB | **198 GB/s** | 9.7 TFLOP/s |
| 4 | 165 µs | 26 MiB | 165 GB/s | 13.0 TFLOP/s |
| 8 | 160 µs | 18 MiB | 118 GB/s | 13.4 TFLOP/s |

Read it in three parts.

**`Mb` 1 → 2: textbook memory-bound behaviour.** Traffic halves, time halves,
and bandwidth doesn't move — it's pinned at ~197 GB/s, the same ceiling lesson
04 hit. When bandwidth is stuck at the ceiling and time tracks traffic exactly,
you are memory-bound, and the only thing that helps is moving fewer bytes.

**`Mb` 2 → 4: the transition.** Traffic drops 38% but time only drops 26%, and
bandwidth falls *off* the ceiling to 165 GB/s. DRAM is no longer the constraint
— you've stopped being able to keep it busy.

**`Mb` 4 → 8: something else is in charge.** Traffic drops another 30%, and time
improves by 3%. Bandwidth collapses to 118 GB/s. Halving the memory traffic
bought essentially nothing, which means memory is no longer what you're waiting
on.

### So what *is* the limit?

The obvious guess is the FPU. Test it — that's what the fidelity cases are for:

| config | time |
|--------|------|
| `Mb=8`, HiFi4 | 159.8 µs |
| `Mb=8`, HiFi2 | 156.1 µs |
| `Mb=8`, LoFi | 155.9 µs |

**A 4× reduction in math passes buys 2%.** So it isn't the multiplier either.

What's left is the *per-instruction* cost of `matmul_tiles`: each call unpacks a
fresh pair of tiles into SrcA/SrcB and issues one 32×32×32 operation, and that
issue-and-unpack overhead — not the mantissa passes — is what the kernel is
paying for. Math fidelity only scales the part that isn't the bottleneck.

The fix is `matmul_block`, which performs an `rt_dim × ct_dim × kt_dim` block in
a single instruction, amortising the unpack across many more MACs. That's how
production matmuls reach a regime where fidelity is worth tuning — and it's
where you'd go next.

### The transferable habit

Every optimisation you make either moves the bottleneck or wastes your time, and
the only way to tell which is to measure after each change. The `Mb=4 → 8` step
looks like progress on paper — a 30% traffic reduction! — and delivered 3%.

Before optimising anything, ask what you're waiting on, change one thing, and
check whether the number that was pinned came unpinned.
