# 04 — Multi-core: making it 64× wider

**Goal:** the same `c = a + b`, spread across the whole 8×8 grid of Tensix
cores. This is the single largest performance change in the whole course, and
it barely touches the kernel.

> **Background:** [`theory 07 — Many cores`](../../theory/07-multi-core.md), and
> [`theory 01 §8`](../../theory/01-latency-and-throughput.md) on why parallelism
> stops helping. [`theory 08`](../../theory/08-performance.md) explains how the
> benchmark numbers are produced.

---

## Theory

### One program, many cores

A tt-metal program places kernels on a **`CoreRangeSet`** — a set of rectangular
regions of the core grid. Every core in that set runs the *same compiled
binary*. What differs between them is their **runtime args**.

So parallelising is not a kernel rewrite. It's:

1. Host side: decide which tiles each core owns, and pass that as runtime args.
2. Kernel side: start the loop at `start_tile_id` instead of 0.

That's the whole change. Your lesson-03 kernel already loops over tile indices;
it just assumed the range began at zero.

```cpp
const uint32_t start_tile = get_arg_val<uint32_t>(...);
const uint32_t n_tiles    = get_arg_val<uint32_t>(...);

for (uint32_t i = start_tile; i < start_tile + n_tiles; i++) {
    ...
}
```

### Splitting the work

With `T` tiles and `C` cores, core `k` gets `T/C` tiles, and the first `T mod C`
cores take one extra. The harness does this for you (`harness.split_tiles`), but
two details are worth internalising:

- **Cores with no work must not run the kernel at all.** If `T < C` and you
  place the kernel on all `C` cores but only give runtime args to `T` of them,
  the rest execute with uninitialised args — usually a hang, sometimes a
  corrupted DRAM write. The harness's `cores_used()` computes the subset that
  actually receives work, and the kernels are placed only there.

- **Imbalance costs you.** The program finishes when the *slowest* core
  finishes. 65 tiles over 64 cores takes as long as 128 tiles over 64 cores: one
  core does 2 tiles while 63 do 1 and then idle. This quantisation is why real
  ops care about picking grid shapes that divide the work evenly.

### Circular buffers are per-core

`harness.cb(...)` takes a core range set, and allocates that CB **on every core
in the set** — each core gets its own private L1 ring buffer at the same
address. There is no sharing. Cores in this exercise never talk to each other;
they each independently stream their own slice of DRAM.

### What limits the speedup

Going from 1 core to 64 will not give you 64×. Element-wise add is
**memory-bound**: per output tile it moves 3 tiles across the NoC (two in, one
out = 6 KB) and does 1024 additions. The FPU can do that arithmetic in a rounding
error of the time it takes to fetch the operands.

So as you add cores you climb steeply at first — one core cannot saturate DRAM —
and then flatten hard when the aggregate DRAM bandwidth becomes the ceiling.
Finding that knee is the point of this exercise.

`./dojo bench 04` sweeps 1 → 2 → 8 → 32 → 64 cores at a fixed problem size and
prints the achieved bandwidth for each. **Look at the GB/s column, not the
microseconds** — that's the number that tells you whether you're near the
hardware limit or leaving performance on the table.

---

## Your task

Adapt your lesson-03 kernels to take a starting tile index. All three kernels
need it — the reader, the compute kernel, and the writer.

- **`kernels/reader.cpp`** — read tiles `[start, start + n)` of `a` and `b`.
- **`kernels/compute.cpp`** — process `n` tiles.
- **`kernels/writer.cpp`** — write tiles `[start, start + n)` of `c`.

> The compute kernel never touches DRAM, so it doesn't care *which* tiles it is
> processing — only how many. Only the two data movement kernels need `start`.

### What the host gives you

**`reader.cpp`**

| | |
|---|---|
| compile-time args | CB `a`, CB `b`, then accessor args for `a` and `b` |
| runtime arg 0 | `a` base address |
| runtime arg 1 | `b` base address |
| runtime arg 2 | number of tiles **for this core** |
| runtime arg 3 | first tile index for this core |

**`compute.cpp`**

| | |
|---|---|
| compile-time args | CB `a`, CB `b`, CB out |
| runtime arg 0 | number of tiles for this core |

**`writer.cpp`**

| | |
|---|---|
| compile-time args | CB out, then accessor args for `c` |
| runtime arg 0 | `c` base address |
| runtime arg 1 | number of tiles for this core |
| runtime arg 2 | first tile index for this core |

### Run it

```bash
./dojo test 04
./dojo bench 04      # the scaling sweep
```

---

## Hints

<details>
<summary>Starting the loop</summary>

```cpp
const uint32_t end_tile = start_tile + n_tiles;
for (uint32_t i = start_tile; i < end_tile; i++) {
    ...
}
```

Recomputing `start_tile + n_tiles` in the loop condition is also fine — the
compiler hoists it — but writing it out makes the intent obvious.

</details>

<details>
<summary>Only some cores produce correct output</summary>

The most likely cause is the compute kernel looping over the wrong count, or a
kernel using the global tile index where it wanted a CB-relative one. Inside a
CB, the tile index is always relative to the current window: with one tile
visible it is `0`, never `i`.

</details>

<details>
<summary>It passes at 64 tiles but hangs at 1000</summary>

Check that all three kernels agree on the per-core tile count. The reader
pushing more tiles than the compute kernel pops will fill the CB and stall; the
reverse will stall waiting for tiles that never come.

</details>

---

## Reading the benchmark

Measured on a Wormhole n150, 2048 tiles:

| cores | time | bandwidth | vs 1 core |
|-------|------|-----------|-----------|
| 1 | 1293 µs | 9.7 GB/s | 1.0× |
| 2 | 647 µs | 19.4 GB/s | 2.0× |
| 8 | 170 µs | 74 GB/s | 7.6× |
| 32 | 64.6 µs | 195 GB/s | 20× |
| 64 | 69.6 µs | 181 GB/s | 18.6× |

Three things to take from this:

1. **Scaling is perfectly linear at first** — 2 cores is exactly 2×, 8 cores is
   7.6×. One core cannot come close to saturating DRAM.

2. **It stops at ~195 GB/s.** That's this chip's practical DRAM ceiling for this
   access pattern. Once you're there, no amount of extra cores helps.

3. **64 cores is *slower* than 32.** Past the knee the extra cores don't just
   fail to help, they actively contend for the same memory system. More
   parallelism is not free, and "use the whole grid" is not automatically right.

Note also how the host time behaves: at 64 cores the device does the work in
70 µs but the host round trip is 168 µs. Dispatch overhead is now larger than
the kernel. That's why the dojo reports device time — and it's a real
consideration when deciding whether an op is worth splitting at all.

Lesson 05 attacks the other axis: making each individual core faster.
