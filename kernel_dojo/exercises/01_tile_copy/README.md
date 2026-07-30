# 01 — Tile copy

**Goal:** move a tensor from DRAM to DRAM through a Tensix core's L1, one tile
at a time. No maths. Just data movement — which is where most of the difficulty
in Tensix programming actually lives.

> **Before you start**, if you haven't already:
> [`theory 00`](../../theory/00-what-is-a-kernel.md) (what a kernel is) and
> [`theory 01`](../../theory/01-latency-and-throughput.md) (latency, async,
> deadlock). This lesson also draws on
> [`02 The chip`](../../theory/02-the-chip.md),
> [`03 Tiles`](../../theory/03-tiles-and-numbers.md),
> [`04 Circular buffers`](../../theory/04-circular-buffers.md) and
> [`05 Data movement`](../../theory/05-data-movement.md) — read them properly,
> or use the recap below and refer back when something is unclear.

---

## Recap

**A Tensix core is five small processors sharing 1464 KB of local memory (L1).**
Two of them do data movement (BRISC and NCRISC); three drive the math engines.
You write a separate program for each one you use. Here you write two: a
*reader* on NCRISC and a *writer* on BRISC.

**There is no cache.** Nothing moves between DRAM and L1 unless a kernel
explicitly asks for it.

**The unit of data is a 32×32 tile** — 2048 bytes in bfloat16. A tensor in
`TILE_LAYOUT` is a sequence of tiles, so kernels loop over tile indices. Tile
`(r, c)` of a matrix `Ct` tiles wide is at page index `r * Ct + c`.

**A circular buffer (CB) is a queue in L1** that the two kernels use to hand
tiles to each other. Four operations, two per side:

| Producer | Consumer |
|---|---|
| `cb_reserve_back(cb, n)` — wait for `n` free slots | `cb_wait_front(cb, n)` — wait for `n` filled slots |
| `cb_push_back(cb, n)` — publish them | `cb_pop_front(cb, n)` — release them |

Between reserve and push the producer writes at `get_write_ptr(cb)`; between
wait and pop the consumer reads from `get_read_ptr(cb)`.

Both `wait`-flavoured calls **block**. If the counts don't reconcile you get a
**deadlock** — the program stops and produces nothing, with no error. That is
the most common bug in this course.

**NoC transfers are asynchronous.** `noc_async_read_page` only *issues* the
request and returns immediately; the data is not in L1 yet. `noc_async_read_barrier()`
waits for every read this processor has outstanding. Touch the data before the
barrier and you silently read stale memory.

**`TensorAccessor` handles DRAM addressing.** Tiles are interleaved across 6
DRAM banks, so "which bank holds page 37" is not simple arithmetic:

```cpp
constexpr auto src_args = TensorAccessorArgs<1>();   // decode CT args from index 1
const auto src = TensorAccessor(src_args, src_addr);
noc_async_read_page(tile_id, src, l1_addr);
```

---

## Your task

Write the two data movement kernels in `kernels/`:

- **`reader.cpp`** — runs on NCRISC. For each tile, read it from DRAM into CB 0.
- **`writer.cpp`** — runs on BRISC. For each tile, take it from CB 0 and write
  it to DRAM.

Single core. The tensor is `[1, 1, 32, 32·N]` in `bfloat16`, so N tiles.

### What the host gives you

**`reader.cpp`**

| | |
|---|---|
| compile-time arg 0 | CB index to write into |
| compile-time args 1.. | `TensorAccessorArgs` for the source |
| runtime arg 0 | source DRAM base address |
| runtime arg 1 | number of tiles |

**`writer.cpp`**

| | |
|---|---|
| compile-time arg 0 | CB index to read from |
| compile-time args 1.. | `TensorAccessorArgs` for the destination |
| runtime arg 0 | destination DRAM base address |
| runtime arg 1 | number of tiles |

### API you need

```cpp
uint32_t get_arg_val<uint32_t>(int idx);          // runtime arg
constexpr auto x = get_compile_time_arg_val(idx); // compile-time arg

cb_reserve_back(cb, n);  cb_push_back(cb, n);     // producer
cb_wait_front(cb, n);    cb_pop_front(cb, n);     // consumer
uint32_t get_write_ptr(cb);   uint32_t get_read_ptr(cb);

noc_async_read_page(page_idx, accessor, dst_l1_addr);
noc_async_write_page(page_idx, accessor, src_l1_addr);
noc_async_read_barrier();
noc_async_write_barrier();
```

### Run it

```bash
./dojo test 01
```

---

## Hints

<details>
<summary>The reader loop, in words</summary>

For each tile index `i` from 0 to `n_tiles`:
1. Reserve one page of space in the CB.
2. Ask where that space is (`get_write_ptr`).
3. Issue the read of page `i` into it.
4. Barrier — wait for the data to actually land.
5. Push the page so the writer can see it.

</details>

<details>
<summary>Why does my test hang?</summary>

The classic causes:

- The reader pushes but the writer never pops → the CB fills and
  `cb_reserve_back` blocks forever.
- Mismatched counts: reserving 1 but pushing 2, or waiting for 2 when only 1
  will ever arrive.
- The two kernels disagree on the number of tiles.

Both kernels must go around the loop exactly `n_tiles` times.

</details>

<details>
<summary>Why is my output garbage / all zeros?</summary>

Almost certainly a missing `noc_async_read_barrier()` before `cb_push_back`, or
a missing `noc_async_write_barrier()` before `cb_pop_front`. Without the write
barrier you free the L1 page while the NoC is still reading out of it, so the
data on its way to DRAM is whatever overwrote it.

</details>

---

## Going further

Once it passes, try `./dojo bench 01` and note the number. It will be
unimpressive — a single core reading 2 KB at a time, with a full round trip
stall on every tile. Lessons 04 and 05 are about fixing exactly that.
