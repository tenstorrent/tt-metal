# 10 — `TensorAccessor` in depth

*A reference appendix, not part of the linear course. Chapter 05 tells you
enough to write the exercises; this explains what is actually happening, why the
API is split into two objects, and what the compile-time arguments contain.*

---

## 1. The problem being solved

You want to read page 37 of a tensor. What address is that?

For a tensor **interleaved across DRAM**, page 37 does not live at
`base + 37 * page_size`. Wormhole has **12 DRAM banks** (6 channels × 2 views of
1 GB each), and consecutive pages are handed out round-robin across them:

```
page:  0   1   2   3   4  ...  11  12  13  14 ...
bank:  0   1   2   3   4  ...  11   0   1   2 ...
```

So page 37 is in bank `37 % 12 = 1`, and it's the `37 / 12 = 3`rd page *within
that bank*. Then you need that bank's physical position on the NoC grid, plus
the bank's own base offset in the address map, plus alignment padding.

The actual computation, from `dataflow_api_addrgen.h`:

```cpp
bank_offset_index = page_id / NUM_DRAM_BANKS;          // which page within the bank
bank_index        = page_id % NUM_DRAM_BANKS;          // which bank
addr = bank_offset_index * align(page_size, allocator_alignment)
     + bank_base_address                                // where this tensor starts
     + offset                                           // caller's extra offset
     + bank_to_dram_offset[bank_index];                 // bank's own base
noc_xy = get_noc_xy(bank_index);                        // bank's grid coordinates
noc_addr = (noc_xy << 32) | addr;                       // 64-bit NoC address
```

Note `NUM_DRAM_BANKS = 12` is **not a power of two**, so the build defines
`IS_NOT_POW2_NUM_DRAM_BANKS` and the division becomes a specialised
constant-divisor multiply rather than a shift. That's the kind of detail you do
not want in your kernel.

And that's only the interleaved case. For a **sharded** tensor the mapping
depends on the tensor's rank, its shape, the shard shape, and the list of banks
the shards were distributed over.

`TensorAccessor` encapsulates all of it behind:

```cpp
noc_async_read_page(37, src, l1_addr);
```

---

## 2. Why two objects

The split is the whole design, and it's worth understanding.

| | What it is | Lives where |
|---|---|---|
| **`TensorAccessorArgs<CTA_OFFSET>`** | A *description of the layout*, decoded from compile-time arguments | entirely in the type system; usually zero bytes |
| **`TensorAccessor`** | That description **+ a runtime base address**, exposing address computation | a few bytes on the stack |

```cpp
constexpr auto src_args = TensorAccessorArgs<1>();      // layout: from CT args at index 1
const auto src = TensorAccessor(src_args, src_addr);    // + base address: runtime
```

The reason for splitting is that **the layout is fixed at compile time but the
address is not**. A buffer gets reallocated every dispatch, so its base address
must be a runtime argument. The layout — interleaved or sharded, DRAM or L1,
page size, shard shape — is known when the program is built, so it can be
baked in as `constexpr`.

That gives you the important property: `TensorAccessorArgs` reads its
configuration through `get_compile_time_arg_val()`, so every branch in the
address computation collapses at compile time. The generated code is as tight as
a hand-written interleaved address generator for that specific layout, without
being hard-coded to it.

In the static case the object genuinely costs nothing: the implementation uses
`[[no_unique_address]]` with `ConditionalField` / `ConditionalStaticInstance`
wrappers so that fields only exist when the corresponding value is *not* a
compile-time constant.

---

## 3. What's actually in the compile-time arguments

### The config word

The **first** compile-time arg at `CTA_OFFSET` is a bitfield
(`tensor_accessor::ArgConfig`, from `hostdevcommon/tensor_accessor/arg_config.hpp`):

| Bit | Flag | Meaning |
|---|---|---|
| 0 | `Sharded` | sharded (vs interleaved) |
| 1 | `IsDram` | in DRAM (vs L1) |
| 2 | `RuntimeRank` | rank comes from a runtime arg |
| 3 | `RuntimeNumBanks` | bank count from a runtime arg |
| 4 | `RuntimeTensorShape` | tensor shape from runtime args |
| 5 | `RuntimeShardShape` | shard shape from runtime args |
| 6 | `RuntimeBankCoords` | bank coordinates from runtime args |
| 7 | `RuntimePageSize` | page size from a runtime arg |

The kernel-side `TensorAccessorArgs` turns each of these into a `static
constexpr bool`, which is what makes the whole thing specialise away.

### The interleaved case — what the dojo uses

For every tensor in this course, the args are **exactly two words**. Measured on
this machine:

```python
>>> ttnn.TensorAccessorArgs(dram_tensor).get_compile_time_args()
[2, 2048]        # config = 0b10 (IsDram, not sharded), aligned_page_size = 2048

>>> ttnn.TensorAccessorArgs(l1_tensor).get_compile_time_args()
[0, 2048]        # config = 0b00 (L1, not sharded)
```

That's it — a flag word and a page size. Everything else the interleaved
formula needs (`NUM_DRAM_BANKS`, `bank_to_dram_offset`, the NoC coordinates) is
already a build-time constant of the kernel, injected by the JIT build.

So `TensorAccessorArgs<1>()` in the dojo's readers is consuming args 1 and 2,
and `next_compile_time_args_offset()` returns 3.

### The sharded case

Sharded is where the variable length comes from. Layout, in order, from
`tensor_accessor_args.h`:

| Words | Contents |
|---|---|
| 1 | config word |
| 1 | `aligned_page_size` |
| 1 | `rank` (omitted if `RuntimeRank`) |
| 1 | `num_banks`, packed (omitted if `RuntimeNumBanks`) |
| `rank` | tensor shape (omitted if `RuntimeTensorShape`) |
| `rank` | shard shape (omitted if `RuntimeShardShape`) |
| `⌈num_banks/2⌉` | bank coordinates, two packed per word (omitted if `RuntimeBankCoords`) |

A rank-4 tensor sharded over 8 banks therefore costs
`1 + 1 + 1 + 1 + 4 + 4 + 4 = 16` compile-time args, versus 2 for interleaved.

Two packing tricks worth knowing:

- **Bank coordinates are two per word**: `((x0 & 0xFF) << 8) | (y0 & 0xFF)` in
  the low half, the next bank in the high half. Hence `⌈num_banks/2⌉`.
- **`num_banks` carries a flag in bit 31.** `pack_num_banks(n, contiguous)` sets
  `1 << 31` to select shard-contiguous (`CONTIGUOUS_1D`) distribution rather
  than round-robin. Bank counts are tiny so the bit is free, and it means the
  distribution strategy costs no extra argument slot.

---

## 4. Why `next_compile_time_args_offset()` exists

Because of section 3: **the number of args consumed is not fixed.** It's 2 for
interleaved, 16 for the sharded example above, fewer if some fields are runtime.

So when a kernel has two tensors, the second accessor must begin where the first
ended:

```cpp
constexpr auto a_args = TensorAccessorArgs<2>();
const auto a = TensorAccessor(a_args, a_addr);

constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
const auto b = TensorAccessor(b_args, b_addr);
```

`next_compile_time_args_offset()` is just `CTA_OFFSET + num_compile_time_args()`,
and `num_compile_time_args()` is the `NumArgsCT` constant computed from the
config word.

**Hard-coding `TensorAccessorArgs<4>()` for the second tensor works** — right up
until someone shards an input, at which point the offset is wrong, the second
accessor decodes a config word that is actually part of the first tensor's shape
data, and you get silent garbage. This is exactly the sort of bug that survives
code review, so use the accessor.

For many tensors there's a helper that does the chaining for you:

```cpp
constexpr auto accessors = make_tensor_accessor_args_tuple<NUM_TENSORS, CTA_OFFSET>();
```

which returns a `std::tuple<TensorAccessorArgs<off0>, TensorAccessorArgs<off1>, ...>`
with the offsets computed recursively at compile time.

---

## 5. `aligned_page_size`, not page size

The second word is the *aligned* page size, and the distinction matters.

The allocator places pages on an alignment boundary (32 bytes for DRAM on
Wormhole, 16 for L1). If your logical page size isn't a multiple of that, each
page occupies slightly more space than it contains, and the stride between pages
is the aligned figure:

```cpp
bank_offset_index * align_power_of_2(page_size, allocator_alignment)
```

For the dojo it's moot — a bfloat16 tile is 2048 bytes, already aligned. But for
a row-major tensor whose row is, say, 100 bytes, the stride is 128 in DRAM, and
using 100 would drift progressively out of alignment.

`noc_async_read_page` picks this up automatically. It inspects the accessor type
and prefers `get_aligned_page_size()`:

```cpp
if constexpr (has_get_aligned_page_size_v<AddrGen>) page_size = addrgen.get_aligned_page_size();
else if constexpr (has_page_size_v<AddrGen>)        page_size = addrgen.page_size;
else                                                page_size = 1 << addrgen.log_base_2_of_page_size;
```

Which is why you never pass a size to `noc_async_read_page` — the transfer size
comes from the accessor.

---

## 6. Runtime-configurable fields (CRTA)

The `Runtime*` flags exist to avoid recompilation.

Kernels are JIT-compiled and cached, keyed on their compile-time args. If a
tensor's shape is a compile-time arg, **every new shape recompiles the kernel** —
seconds of latency, and cache bloat. For an op that runs on many shapes that's
unacceptable.

Setting `RuntimeTensorShape` moves the shape out of the compile-time args and
into **common runtime args** (CRTA) — runtime args shared by all cores, rather
than per-core. The kernel then reads it with `get_common_arg_val<uint32_t>(...)`
instead of `get_compile_time_arg_val(...)`, and one compiled binary serves every
shape.

The two-parameter form declares where that CRTA section starts:

```cpp
TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>
```

`CRTA_OFFSET` defaults to `tensor_accessor::UNKNOWN`, in which case you pass it
at construction: `TensorAccessorArgs<CTA_OFFSET>(crta_offset)`. There's a
matching `next_common_runtime_args_offset()` for chaining.

The trade is the obvious one: runtime fields cost a load and defeat constant
folding in the address computation. Compile-time is faster per access; runtime
avoids recompiles. Interleaved tensors in a fixed pipeline want compile-time
(what the dojo does); a general-purpose op over arbitrary shapes wants runtime.

### Constraints the header enforces

Three `static_assert`s catch impossible combinations:

- **Runtime rank requires runtime tensor and shard shapes.** Their arg counts
  depend on the rank, so with a runtime rank you cannot compute the offsets at
  compile time.
- **Runtime `num_banks` requires runtime bank coords**, for the same reason.
- **`RuntimePageSize` is not allowed on sharded tensors.** It exists for
  interleaved row-major tensors with a dynamic shape, where
  `page_size = last_dim_width × element_size`. Sharded and interleaved-tiled
  tensors always have a static page size.

---

## 7. The sharded addressing path

For completeness — the dojo never uses it, but this is what the extra args buy.

A sharded tensor is described by a **DistributionSpec** (`dspec`): tensor shape,
shard shape, and the bank list. `get_bank_and_offset` then does, per dimension:

```cpp
for (i = 0; i < rank; ++i) {
    flattened_shard_id      += (page_coord[i] / shard_shape[i]) * shard_grid_strides[i];
    page_offset_within_shard += (page_coord[i] % shard_shape[i]) * shard_strides[i];
}
bank_shard = shard_to_bank(flattened_shard_id);
bank_page_offset = bank_shard.shard_in_bank * shard_volume + page_offset_within_shard;
```

Integer-divide the coordinate by the shard shape to find *which* shard; take the
remainder to find *where in* the shard. `shard_to_bank` then maps shard → bank,
round-robin or contiguous depending on that bit-31 flag.

Note the `rank >= 4` shortcut in `get_bank_and_offset(page_id)`: for rank 4 and
above it uses a direct page-id path rather than decomposing into coordinates,
avoiding the loop.

Sharded accessors also expose APIs the interleaved specialisation deliberately
`static_assert`s against — `is_local_bank`, `is_local_page`, `shard_pages`,
`strided_shard_pages`. `is_local_*` is how a kernel checks whether data is
already in its own L1 and skips the NoC entirely, which is the point of
sharding.

---

## 8. Using it directly

`noc_async_read_page` is a convenience wrapper. The accessor is usable on its
own:

```cpp
uint64_t addr = src.get_noc_addr(page_id);           // 64-bit NoC address
uint64_t addr = src.get_noc_addr(page_id, offset);   // with a byte offset

// then, e.g.
noc_async_read(addr, l1_dst, size_in_bytes);
```

This is what you'd reach for to read *part* of a page, to read with a custom
transfer size, or to compute an address once and reuse it via the
`noc_async_read_*_with_state` family.

There is also a **`TensorBindingToken`** constructor (the "Metal 2.0" path),
which bundles the base address into the CRTA section so the accessor can be
built from a single token, with the framework patching the address on each
dispatch. You'll see it in newer ttnn ops.

---

## 9. The host side

In Python, as the dojo's harness does:

```python
args = list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
```

and in `harness.py`:

```python
def accessor_args(tensor) -> list[int]:
    return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
```

These words go into the kernel's `compile_time_args` at the position the kernel
expects. **The host and the kernel must agree on that position** — there is no
checking. If the kernel says `TensorAccessorArgs<2>()`, the host must have put
the accessor words at index 2, which is why the dojo's task files build the list
in a fixed order:

```python
ct_args=[CB_A, CB_B, *harness.accessor_args(a), *harness.accessor_args(b)]
#         0     1     2...                       then wherever a's ended
```

In C++ the equivalent is `TensorAccessorArgs(buffer).append_to(compile_time_args)`,
with a two-argument overload that also appends the common runtime args when any
`Runtime*` flag is set.

---

## 10. Pitfalls

| Symptom | Cause |
|---|---|
| Second tensor reads garbage | Hard-coded CT offset instead of `next_compile_time_args_offset()` |
| All reads garbage, tensor is in L1 | `IsDram` mismatch — accessor built from the wrong tensor's args |
| Works for interleaved, breaks when sharded | Assumed a fixed arg count somewhere |
| Off-by-a-few addressing on row-major data | Used logical page size instead of the aligned one |
| Kernel recompiles constantly | Shape is a compile-time arg; consider `RuntimeTensorShape` |
| Reads the right data from the wrong tensor | Host built `ct_args` in a different order than the kernel decodes |

The recurring theme: the compile-time args are a **positional, untyped
contract** between host and kernel. Nothing validates it. Both sides must agree,
and the accessor's offset helpers exist so that agreement survives a change of
memory layout.

---

## Source

| | |
|---|---|
| `tt_metal/hw/inc/api/tensor/tensor_accessor_args.h` | the args decoder, offset chain, `static_assert`s |
| `tt_metal/hw/inc/api/tensor/tensor_accessor.h` | the accessor, sharded and interleaved specialisations |
| `tt_metal/hw/inc/internal/tensor/dspec.h` | DistributionSpec |
| `tt_metal/hostdevcommon/api/hostdevcommon/tensor_accessor/arg_config.hpp` | the config bits, `pack_num_banks` |
| `tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h` | `InterleavedAddrGen`, the interleaved formula |
| `tt_metal/impl/buffers/tensor_accessor_args.cpp` | host-side arg generation |

---

**Back to:** [05 — Data movement](05-data-movement.md) ·
[the theory index](../THEORY.md)
