# 05 — Pipelining: making one core fast

**Goal:** same add, same core count, but 2.6× faster. Lesson 04 made the kernel
*wider*; this makes it *deeper*. The technique is blocking, and it is the most
transferable performance idea in the course.

> **Background — read this first if the words are unfamiliar.**
> [`theory 01`](../../theory/01-latency-and-throughput.md) sections 2–5 define
> everything this lesson uses: *in flight*, *batching*, *pipelining*, and
> *double buffering*, all from scratch and without reference to Tenstorrent
> hardware. This lesson is the applied version of that chapter.

---

## Theory

### Your lesson-04 kernel is a stop-and-go pipeline

Look at what the reader actually does per tile:

```
issue read ──▶ [ wait ~hundreds of cycles for DRAM ] ──▶ push ──▶ issue read ──▶ [ wait ] ──▶ ...
```

`noc_async_read_barrier()` drains *everything outstanding*, and you call it with
exactly one read in flight. So every tile pays the full DRAM round-trip latency,
serially. Meanwhile the compute pipeline is idle waiting for that tile, and once
it gets one, the reader is idle waiting for compute to free the page.

Nothing overlaps. Three processors, one of them working at a time.

### Fix 1 — batch the reads

Issue several reads *before* barriering:

```cpp
for (uint32_t t = 0; t < block; t++) {
    noc_async_read_page(i + t, a, addr_a + t * tile_bytes);
    noc_async_read_page(i + t, b, addr_b + t * tile_bytes);
}
noc_async_read_barrier();       // one wait for 2*block transactions
```

Now `2 × block` transactions are in flight at once and their latencies overlap.
The NoC is built to have many outstanding requests — one at a time is the one
case it handles badly.

Note the address arithmetic: after `cb_reserve_back(cb, block)`,
`get_write_ptr(cb)` gives the start of a **contiguous run of `block` pages**, so
page `t` sits at `get_write_ptr(cb) + t * tile_size`. Get the tile size with
`get_tile_size(cb)`.

### Fix 2 — batch the DST handshake

The `tile_regs_acquire / commit / wait / release` quartet is a synchronisation
between the math and pack threads, and it isn't free. Per tile, it can rival the
cost of the add itself.

DST holds **8 tiles** in the default half-sync mode (16 physically, but math and
pack work on opposite halves so they can overlap; `fp32_dest_acc_en` halves it
again to 4). So amortise the handshake across a whole block:

```cpp
tile_regs_acquire();
for (uint32_t t = 0; t < block; t++) {
    add_tiles(cb_a, cb_b, t, t, t);      // CB index t, DST slot t
}
tile_regs_commit();

tile_regs_wait();
for (uint32_t t = 0; t < block; t++) {
    pack_tile(t, cb_out);                 // packs sequentially into the CB
}
tile_regs_release();
```

Two things changed. The handshake happens once per block instead of once per
tile. And `add_tiles` now indexes into the CB: after `cb_wait_front(cb_a, block)`
the visible window is `block` tiles wide, indexed `0..block-1`.

### Fix 3 — give the CB room to breathe

A CB with 1 page cannot overlap anything. The producer fills the only page, then
must wait for the consumer to empty it before writing again; the consumer then
waits for the producer to refill it. They take turns, and only one of them is
ever working — which throws away everything the pipeline was for.

Give it two pages and the producer fills page B while the consumer drains page
A, then they swap. Both work continuously. That is **double buffering**.

More pages tolerate more variation in timing: if the reader is occasionally
slow, a deeper buffer means compute has more banked up before it starves. The
cost is L1 space. `harness.cb(...)` takes an `n_pages` argument for exactly this,
and in this exercise the host sizes every CB at `2 × block`.

### The costs

Blocking is not free:

- **L1 space.** Three CBs × 2 blocks × 2 KB. At `block = 8` that's 96 KB of the
  core's ~1.5 MB. Fine here; for a matmul with large blocks it becomes the
  binding constraint.
- **DST pressure.** More than 8 tiles per block simply won't fit.
- **Granularity.** The tile count per core must be a multiple of the block size,
  or you need a remainder path. This exercise guarantees divisibility so you can
  focus on the pipeline.

### What to expect

`./dojo bench 05` sweeps block sizes 1, 2, 4, 8 at a fixed 8 cores. Block size 1
is essentially your lesson-04 kernel. Measured on a Wormhole n150, 2048 tiles:

| block | time | bandwidth |
|-------|------|-----------|
| 1 | 170 µs | 74 GB/s |
| 2 | 109 µs | 115 GB/s |
| 4 | 75.8 µs | 166 GB/s |
| 8 | 64.6 µs | 195 GB/s |

**2.6× faster on exactly the same eight cores.** Nothing was parallelised; the
existing hardware was simply kept busy instead of idling on round trips.

Compare that 64.6 µs with lesson 04's table: it is the *same* number 32
unblocked cores achieved. Blocking bought you a 4× reduction in cores for the
same throughput — and it lands you on the same ~195 GB/s DRAM ceiling, which is
the real limit for this op however you get there.

---

## Your task

Rewrite the reader and compute kernels to work a block at a time.

- **`kernels/reader.cpp`** — for each block, reserve `block` pages in each CB,
  issue all `2 × block` reads, one barrier, push `block` pages to each.
- **`kernels/compute.cpp`** — for each block, wait for `block` tiles, do the
  whole block inside one DST acquire/commit, pack all `block` results.
- **`kernels/writer.cpp`** — provided, already blocked. Read it: it shows the
  same batching idea on the write side.

`block` arrives as a **compile-time** arg, so the loops unroll and the address
arithmetic folds into constants.

### What the host gives you

**`reader.cpp`**

| | |
|---|---|
| compile-time arg 0 | CB `a` |
| compile-time arg 1 | CB `b` |
| compile-time arg 2 | `block` — tiles per block |
| compile-time args 3.. | accessor args for `a`, then `b` |
| runtime args | `a` addr, `b` addr, tiles for this core, first tile index |

**`compute.cpp`**

| | |
|---|---|
| compile-time args | CB `a`, CB `b`, CB out, `block` |
| runtime arg 0 | tiles for this core (always a multiple of `block`) |

### API you need

```cpp
uint32_t get_tile_size(cb);     // bytes per page, for address arithmetic
```

everything else you have already used.

### Run it

```bash
./dojo test 05
./dojo bench 05
```

---

## Hints

<details>
<summary>Reader block loop</summary>

```cpp
const uint32_t tile_bytes = get_tile_size(cb_a);

for (uint32_t i = start; i < end; i += block) {
    cb_reserve_back(cb_a, block);
    cb_reserve_back(cb_b, block);

    const uint32_t base_a = get_write_ptr(cb_a);
    const uint32_t base_b = get_write_ptr(cb_b);

    for (uint32_t t = 0; t < block; t++) {
        noc_async_read_page(i + t, a, base_a + t * tile_bytes);
        noc_async_read_page(i + t, b, base_b + t * tile_bytes);
    }
    noc_async_read_barrier();

    cb_push_back(cb_a, block);
    cb_push_back(cb_b, block);
}
```

Call `get_write_ptr` **once per block**, before the inner loop — it returns the
start of the reserved run, and it does not advance as you write.

</details>

<details>
<summary>Wrong results at block > 1, correct at block == 1</summary>

Two usual suspects:

- `add_tiles(cb_a, cb_b, 0, 0, t)` — using CB index 0 for every tile in the
  block, so you add the first tile `block` times. The CB index must be `t`.
- Forgetting to scale the address by `t * tile_bytes`, so every read of the
  block lands on the same page.

</details>

<details>
<summary>It hangs at block == 8</summary>

Check the CB has room for a whole block: `cb_reserve_back(cb, 8)` on a CB with
fewer than 8 pages can never succeed. The host sizes CBs at `2 × block` here, so
if you're seeing this you're probably reserving more than you think.

</details>

---

## Going further

- Push `block` to 8 and add lesson 04's full 64-core grid. Do the two
  optimisations compose, or does one absorb the other's gains? (The answer tells
  you which resource you're actually short of.)
- Try decoupling the read block from the compute block — read 8, compute 4.
  Does the extra CB depth help beyond what `2 × block` already gave you?
