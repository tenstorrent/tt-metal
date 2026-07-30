# 07 — Many cores

*Using more than one of the 64, and what happens when they need to cooperate.*

Prerequisite: [chapter 01, section 8](01-latency-and-throughput.md) — why
parallelism stops helping.

---

## Same binary, different arguments

A kernel is launched onto a **set of cores**. Every core in that set runs the
*same compiled binary*. What makes them do different work is their **runtime
arguments**, which are set per core.

So parallelising a kernel is usually not a kernel rewrite. It's:

1. **Host side:** decide which slice of the data each core owns, and pass that
   as runtime args.
2. **Kernel side:** start the loop at the given offset instead of 0.

```cpp
const uint32_t start_tile = get_arg_val<uint32_t>(2);
const uint32_t n_tiles    = get_arg_val<uint32_t>(3);

const uint32_t end_tile = start_tile + n_tiles;
for (uint32_t i = start_tile; i < end_tile; i++) {
    ...
}
```

That's the entire change from lesson 03 to lesson 04, and it's worth a 20×
speedup.

Note that a **compute** kernel usually doesn't need the offset at all — it only
sees circular buffers, so it just needs to know *how many* tiles to process, not
which ones. Only the kernels touching DRAM care about position.

---

## Splitting the work

With `T` items and `C` cores, the usual split gives each core `T/C` items, and
the first `T mod C` cores take one extra. Two things to get right.

### Cores with no work must not run the kernel

If `T < C`, some cores get nothing. If you place the kernel on them anyway, they
execute with **uninitialised runtime arguments** — reading whatever was in that
memory. Usually a hang; sometimes a write to a garbage DRAM address, which
corrupts an unrelated buffer.

The dojo's harness computes the subset of cores that actually receive work
(`cores_used`) and places kernels only there.

### Imbalance is pure waste

The program finishes when the **slowest** core finishes. 65 tiles over 64 cores
takes exactly as long as 128 tiles over 64 cores: one core does 2 while 63 do 1
and then idle.

This quantisation is why real kernels care about picking grid shapes that divide
the work evenly, and why the benchmarks in this course use sizes like 2048 and
Mt=64 rather than round decimal numbers.

---

## Circular buffers are per-core

When the host declares a CB on a core range, **each core gets its own private
copy** in its own L1, at the same address. There is no sharing.

In lessons 04–08 the cores never communicate. Each independently streams its own
slice of DRAM through its own CBs. That's the easy case, and it covers a large
fraction of real operations.

---

## When cores do need to cooperate

Three mechanisms, none of which this course's exercises require, but all of
which you'll meet immediately afterwards.

### Semaphores

A **semaphore** is a small counter in L1 that cores can increment remotely over
the NoC, and wait on locally.

```cpp
noc_semaphore_wait(sem_addr, expected_value);   // block until it reaches this
noc_semaphore_inc(remote_addr, 1);              // bump another core's counter
```

That's the primitive for "core A tells core B that data is ready". It's the
cross-core equivalent of the circular buffer's push/pop, but manual — you're
responsible for the counting.

### Multicast

When many cores need the *same* data — very common in matmul, where a whole row
of cores needs the same operand — having each core read it from DRAM
independently wastes bandwidth proportional to the number of cores.

**Multicast** sends one NoC transaction to a whole rectangle of cores:

```cpp
noc_async_write_multicast(src_l1_addr, multicast_dest_addr, size, num_dests);
```

One core fetches from DRAM once, then broadcasts to its neighbours. On an 8-wide
grid that's an 8× reduction in traffic for the shared operand. This is the
single biggest optimisation in production matmul kernels, and it's the natural
next step after lesson 08.

### Sharding

Instead of interleaving a tensor across DRAM banks, **sharding** places known
slices directly in specific cores' L1 up front. Data starts where it's needed;
the kernel skips the DRAM round trip entirely.

`TensorAccessor` handles the addressing either way, which is the point of the
abstraction — the same kernel body works for interleaved and sharded inputs, and
the layout is described by compile-time args rather than baked into the code.

---

## A worked example in the repo

`tt_metal/programming_examples/matmul/matmul_multicore_reuse_mcast/` uses all
three: semaphores for the handshake, multicast for the shared operands, and
2-D blocking for reuse. It's the natural thing to read after finishing this
course, and lesson 08 leaves off roughly where it begins.

---

## Choosing a core count

The instinct is to use all 64. The measurements say otherwise.

From lesson 04 (element-wise add, 2048 tiles):

| cores | time | bandwidth |
|---|---|---|
| 1 | 1293 µs | 9.7 GB/s |
| 2 | 647 µs | 19.4 GB/s |
| 8 | 170 µs | 74 GB/s |
| 32 | **64.6 µs** | **195 GB/s** |
| 64 | 69.6 µs | 181 GB/s |

Linear to 8, saturated by 32, **worse at 64**. Past the knee, extra cores
contend for a memory system that's already full.

And from lesson 05: 8 cores with a well-pipelined kernel matches 32 cores with a
naive one. Cores are not the only axis, and often not the best one.

The general shape: **parallelism until you saturate a shared resource, then
efficiency**. Knowing which regime you're in requires measuring — chapter 08.

---

**Next:** [08 — Performance](08-performance.md) — measuring, and the full set of
levers with real numbers.
