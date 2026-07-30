# 05 — Data movement

*Getting tiles between DRAM and L1. Half the work of writing kernels, and most
of the performance.*

Prerequisite: [chapter 01, sections 1–3](01-latency-and-throughput.md) —
latency, in-flight requests, batching.

---

## Everything is asynchronous

The two calls you'll use most:

```cpp
noc_async_read_page (page_index, accessor, l1_address);   // DRAM  → L1
noc_async_write_page(page_index, accessor, l1_address);   // L1    → DRAM
```

Both **return immediately**. They put a request on the network and carry on.
When the call returns, *nothing has moved yet*.

To actually wait:

```cpp
noc_async_read_barrier();    // wait for all outstanding reads on this processor
noc_async_write_barrier();   // wait for all outstanding writes
```

### Barriers are per-processor, not per-request

This is the important detail, and it's what makes fast code easy to write.

`noc_async_read_barrier()` does not take an argument identifying which read to
wait for. It waits for **every read this processor has issued and not yet waited
on**. All of them, together.

So this:

```cpp
noc_async_read_page(0, src, addr0);
noc_async_read_page(1, src, addr1);
noc_async_read_page(2, src, addr2);
noc_async_read_page(3, src, addr3);
noc_async_read_barrier();              // one wait for all four
```

costs roughly the same wall-clock time as issuing *one* read and waiting for it,
because the four transfers overlap. Whereas this:

```cpp
noc_async_read_page(0, src, addr0);  noc_async_read_barrier();
noc_async_read_page(1, src, addr1);  noc_async_read_barrier();
noc_async_read_page(2, src, addr2);  noc_async_read_barrier();
noc_async_read_page(3, src, addr3);  noc_async_read_barrier();
```

costs four times as much, for identical data. The NoC is built to have many
requests outstanding; feeding it one at a time is the single most common
performance mistake in a reader kernel.

**Measured, from lesson 05:** moving from 1 tile per barrier to 8 tiles per
barrier makes the kernel **2.6× faster** on unchanged hardware.

### Using data before the barrier

If you read a page and use it without barriering first, you get whatever was in
that L1 memory before — stale data from a previous iteration, or uninitialised
garbage.

There is no error. No crash. Just wrong numbers, and often only *sometimes*
wrong, because whether the data happens to have arrived depends on timing.

The same applies on the write side, in mirror image. If you free a buffer (via
`cb_pop_front`) before the write out of it has completed, the producer will
refill that page while the NoC is still streaming the old contents to DRAM. You
get a mixture.

> **Rule:** a read barrier before you publish the data; a write barrier before
> you release the buffer.

### `noc_async_writes_flushed()`

A weaker write barrier: it waits only until the requests have been *sent*, not
until they've completed. It's cheaper, and it's safe as long as you don't touch
the source buffer.

The common pattern is to flush inside the loop and issue one full
`noc_async_write_barrier()` at the very end of the kernel. Not needed for this
course, but you'll see it in production kernels.

---

## Addressing: `TensorAccessor`

A tensor's tiles are spread across DRAM banks (chapter 02). Working out "which
bank and which offset holds page 37" depends on the number of banks, the page
size, and whether the tensor is interleaved or sharded. You don't want to write
that.

`TensorAccessor` does it:

```cpp
constexpr auto src_args = TensorAccessorArgs<0>();   // decode CT args from index 0
const auto src = TensorAccessor(src_args, src_addr);

noc_async_read_page(37, src, l1_addr);               // handles the rest
```

Two pieces:

- **`TensorAccessorArgs<N>()`** reads *compile-time arguments* starting at index
  `N`. The host packs a description of the memory layout into those args.
- **`TensorAccessor(args, base_address)`** combines that layout with the runtime
  base address of the buffer.

The layout is compile-time, so the address arithmetic gets specialised and
inlined — it's as fast as hand-written code for the specific layout, without
being hard-coded to it.

### Chaining two accessors

Here's the bit that bites. `TensorAccessorArgs<N>` consumes a **variable** number
of compile-time args — how many depends on the memory configuration. So the
second accessor must start where the first one finished:

```cpp
constexpr auto a_args = TensorAccessorArgs<2>();
const auto a = TensorAccessor(a_args, a_addr);

constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
const auto b = TensorAccessor(b_args, b_addr);
```

Never hard-code the second offset. For the dojo's interleaved tensors the args
are exactly two words (`[config, aligned_page_size]`, measured as `[2, 2048]`),
but a sharded rank-4 tensor over 8 banks costs 16 — so a hard-coded offset works
until someone changes the memory config, and then it silently decodes the wrong
words.

> [`Chapter 10`](10-tensor-accessor.md) is a deep dive on this: what the
> compile-time words actually contain, the real interleaved address formula, the
> sharded path, and how to make shapes runtime-configurable so the kernel stops
> recompiling.

---

## Kernel arguments

Kernels get their parameters two ways.

### Compile-time arguments

```cpp
constexpr uint32_t cb_in = get_compile_time_arg_val(0);
```

Baked in when the kernel is compiled. Because they're `constexpr`, the compiler
can unroll loops over them, fold address arithmetic into constants, and drop
dead branches. Use these for anything structural: CB indices, block sizes,
memory layout.

Changing one means recompiling the kernel — which the framework does
automatically, but it costs time and creates a new cache entry.

### Runtime arguments

```cpp
const uint32_t n_tiles = get_arg_val<uint32_t>(0);
```

Passed at launch, **per core**. This is the mechanism that makes 64 cores
running identical code do different work: each gets its own start offset and
count.

Use these for anything that varies per core or per invocation: buffer addresses,
tile ranges, sizes.

---

## The reader/writer pattern

Nearly every data movement kernel in this course is a variation on:

```cpp
void kernel_main() {
    // runtime: what this core is responsible for
    const uint32_t src_addr   = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles    = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);

    // compile-time: structure
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto args = TensorAccessorArgs<1>();
    const auto src = TensorAccessor(args, src_addr);

    const uint32_t end_tile = start_tile + n_tiles;

    for (uint32_t i = start_tile; i < end_tile; i++) {
        cb_reserve_back(cb_out, 1);
        noc_async_read_page(i, src, get_write_ptr(cb_out));
        noc_async_read_barrier();
        cb_push_back(cb_out, 1);
    }
}
```

And the blocked version, which is the same thing done properly:

```cpp
    const uint32_t tile_bytes = get_tile_size(cb_out);

    for (uint32_t i = start_tile; i < end_tile; i += BLOCK) {
        cb_reserve_back(cb_out, BLOCK);
        const uint32_t base = get_write_ptr(cb_out);      // once, outside

        for (uint32_t t = 0; t < BLOCK; t++) {
            noc_async_read_page(i + t, src, base + t * tile_bytes);
        }
        noc_async_read_barrier();                          // one wait for BLOCK

        cb_push_back(cb_out, BLOCK);
    }
```

The difference between those two loops is a factor of 2–3 in throughput.

---

## Beyond this course

Things that exist and matter in production kernels:

- **`noc_async_write_multicast`** — one core sends the same data to a whole
  rectangle of cores in a single transaction. When many cores need the same
  operand, this cuts DRAM traffic by the number of receivers. See chapter 07.
- **Sharding** — instead of interleaving a tensor across DRAM, place known
  slices directly in specific cores' L1 ahead of time. `TensorAccessor` handles
  the addressing either way, so the same kernel body works for both.
- **`noc_async_read_one_packet`** — a lighter-weight path for transfers you know
  fit in a single NoC packet.
- **Explicit NoC selection** — you can choose which of the two networks a kernel
  uses, to balance traffic.

---

**Next:** [06 — Compute](06-compute.md) — the math engines.
