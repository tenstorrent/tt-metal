# 04 — Circular buffers

*The queues that connect your kernels. Everything hangs off understanding these
properly, including most of your bugs.*

Prerequisite: [chapter 01, sections 5–6](01-latency-and-throughput.md) — double
buffering, producers and consumers, deadlock.

---

## What it is

A **circular buffer** (CB) is a fixed-size queue in a core's L1 memory, used to
pass tiles between the processors on that core.

It's "circular" because it's a **ring buffer**: a block of memory with a read
position and a write position that both wrap around to the start when they reach
the end. That way a queue of 4 slots can carry thousands of tiles without ever
moving data or allocating anything.

```
        ┌───┬───┬───┬───┐
        │ 0 │ 1 │ 2 │ 3 │      4 slots ("pages")
        └───┴───┴───┴───┘
              ▲       ▲
              │       └── producer writes here, then wraps to 0
              └────────── consumer reads here
```

Each slot is called a **page**. In this course a page is always exactly one
tile, which keeps the arithmetic simple.

Two numbers define a CB:

- **page size** — bytes per slot (2048 for a bfloat16 tile)
- **number of pages** — the depth of the queue

Depth is the double-buffering knob from chapter 01. `n_pages=1` means producer
and consumer must take turns. `n_pages=2` is double buffering. Deeper tolerates
more timing variation, at the cost of L1.

### Where they live

Each core has **32 CB slots**, numbered 0–31. Conventionally 0–7 are used for
inputs and 16–23 for outputs, but that's just convention — the hardware treats
all 32 identically.

CBs are **per-core and private**. When the host declares a CB on a set of cores,
every core in that set gets its own independent copy at the same address. Cores
never share a CB. (Cores talk to each other by other means — chapter 07.)

---

## The four operations

| Producer side | Consumer side |
|---|---|
| `cb_reserve_back(cb, n)` | `cb_wait_front(cb, n)` |
| `cb_push_back(cb, n)` | `cb_pop_front(cb, n)` |

**Producer:**

```cpp
cb_reserve_back(cb, 1);            // block until 1 page is free
uint32_t addr = get_write_ptr(cb); // where to write it
... put a tile at addr ...
cb_push_back(cb, 1);               // publish it — consumer can now see it
```

**Consumer:**

```cpp
cb_wait_front(cb, 1);              // block until 1 page has data
uint32_t addr = get_read_ptr(cb);  // where to read it
... use the tile at addr ...
cb_pop_front(cb, 1);               // release it — producer can reuse the page
```

`cb_reserve_back` and `cb_wait_front` **block** — they stop the processor until
the condition is satisfied. That's what keeps producer and consumer in step, and
it's what hangs when the counts are wrong.

---

## Three things that catch everyone

### 1. `get_write_ptr` does not advance

It returns the address of the **start of the reserved run**, and it keeps
returning that same address until you push.

So if you reserve 4 pages and want to fill all of them, you do the arithmetic
yourself:

```cpp
cb_reserve_back(cb, 4);
uint32_t base = get_write_ptr(cb);           // call ONCE
uint32_t tile_bytes = get_tile_size(cb);

for (uint32_t t = 0; t < 4; t++) {
    ... write a tile at base + t * tile_bytes ...   // step manually
}
cb_push_back(cb, 4);
```

Calling `get_write_ptr` inside the loop gives you the same address four times,
and all four tiles land on top of each other. This is a very common bug and it
produces "the first tile is right, the rest are duplicates of it".

Reserving `n` pages always gives you `n` **contiguous** pages, so the stride
arithmetic is safe.

### 2. Tile indices are relative to the visible window

When a compute operation takes a "tile index", it means *the index within the
CB's currently visible window* — not the tile's index in your tensor.

```cpp
cb_wait_front(cb, 4);            // 4 tiles now visible: indices 0, 1, 2, 3
add_tiles(cb_a, cb_b, 2, 2, 0);  // uses the THIRD visible tile of each
```

After `cb_pop_front(cb, 4)`, the next four tiles become indices 0–3 again. The
window slides; the indices restart.

So in a loop over tiles `i = 100..200`, the CB index is *not* `i`. If only one
tile is visible it is always `0`.

### 3. You can never wait for more than the CB holds

`cb_reserve_back(cb, 8)` on a CB with 4 pages can **never** succeed. There will
never be 8 free pages, because there are only 4 in total. The kernel blocks
forever.

Whenever you work in blocks of `n`, the CB must have at least `n` pages — and
you want `2 * n` so the producer can prepare the next block while the current
one is consumed.

---

## The accounting rule

> Over the life of the kernel, every `reserve` must have a matching `push`, and
> every `wait` must have a matching `pop` — **and the totals must agree between
> the kernels sharing the buffer.**

They don't have to match per iteration. A producer can push 4 at a time while a
consumer pops 1 at a time; that's fine, and sometimes useful. What matters is
that the totals reconcile and nobody waits for something that will never come.

### Diagnosing a hang

If your test hangs (in the dojo, it fails after 30 seconds with "device hang"),
work through this list:

| Symptom | Likely cause |
|---|---|
| Hangs immediately, first iteration | Waiting for more pages than the CB has |
| Hangs after a few iterations | Reserve/push counts unequal — buffer fills up |
| Hangs at the end | One kernel loops more times than another |
| Hangs only at large sizes | Fine at depth 2 by luck; a real imbalance shows once the buffer fills |

The mechanical check: for each CB, count the total pages pushed by the producer
and the total popped by the consumer, for the whole run. If those two numbers
differ, that's your bug.

---

## A worked pair

The complete lesson-01 pattern — reader and writer sharing one CB:

```cpp
// reader (NCRISC)                    // writer (BRISC)
for (i = 0; i < n; i++) {             for (i = 0; i < n; i++) {
    cb_reserve_back(cb, 1);               cb_wait_front(cb, 1);
    addr = get_write_ptr(cb);             addr = get_read_ptr(cb);

    noc_async_read_page(i, src, addr);    noc_async_write_page(i, dst, addr);
    noc_async_read_barrier();             noc_async_write_barrier();

    cb_push_back(cb, 1);                  cb_pop_front(cb, 1);
}                                     }
```

Both loop `n` times. Both do exactly one reserve/push and one wait/pop per
iteration. The counts reconcile, so it terminates.

Notice the barriers sit *inside* the reserve/push and wait/pop pairs. That
matters:

- The read barrier must come **before** `cb_push_back`, or you publish a page
  whose data hasn't arrived yet.
- The write barrier must come **before** `cb_pop_front`, or you free a page the
  NoC is still reading out of — and the producer will overwrite it mid-flight.

Both mistakes give wrong data with no error message. Chapter 05 goes into why.

---

**Next:** [05 — Data movement](05-data-movement.md) — getting tiles in and out
of DRAM.
