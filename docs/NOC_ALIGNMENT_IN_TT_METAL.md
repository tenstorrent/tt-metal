# NOC Alignment in TT-Metal

This note is a handoff from the TT-Metal runtime and hardware side to ops authors. It describes the alignment rules the hardware imposes, the buffer/accessor contracts the runtime layers on top, and the shape of the problems an op author can expect to run into. It deliberately does not prescribe how any particular op should solve them.

## Short Version

There are two different notions of "page size":

- `page_size`: the logical amount of tensor data in one page/row/stick.
- `aligned_page_size`: the physical stride used when pages are laid out in device memory and when most page-based NOC helpers step through memory.

The important consequence is:

- a page can be logically narrow,
- but if it lives in a device buffer it is usually **spaced in memory** by an aligned stride,
- and any kernel that touches it must either operate on that aligned stride, reject the input, or explicitly bounce through an aligned scratch buffer.

Row-major tensors are the case where this regularly bites, because one page is usually one row/stick and the logical row width is often smaller than the hardware alignment.

## 1. What "Alignment" Means Here

At the hardware level, TT-Metal models different alignment rules depending on where traffic starts and ends.

- On Wormhole, `L1` reads/writes are 16-byte aligned, while `DRAM`/`PCIe` reads are 32-byte aligned.
- On Blackhole, `L1` still uses 16-byte alignment, while `DRAM`/`PCIe` reads are 64-byte aligned.

See:

- `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h:290`
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:374`

The allocator code states the routing rules directly:

- Tensix/Eth <-> Tensix/Eth: `L1_ALIGNMENT`
- Tensix/Eth -> PCIe/DRAM: `L1_ALIGNMENT`
- PCIe/DRAM -> Tensix/Eth: `DRAM_ALIGNMENT`

See:

- `tt_metal/impl/allocator/l1_banking_allocator.cpp:216`

So if a kernel touches DRAM, the DRAM-side alignment matters even when the local buffer only needs 16-byte L1 alignment.

## 2. The Buffer Contract: `page_size` vs `aligned_page_size`

This is the core TT-Metal abstraction that explains most behavior.

- `buffer.page_size()` is the logical page size.
- `buffer.alignment()` is the alignment requirement of that buffer type.
- `buffer.aligned_page_size()` is `align(page_size, alignment)`.

See:

- `tt_metal/impl/buffers/buffer.cpp:552`
- `tt_metal/impl/buffers/buffer.cpp:560`

This means two adjacent logical pages are not necessarily packed back-to-back at `page_size` byte spacing. Interleaved page address calculation uses the aligned stride:

- `page_address()` steps by `round_up(page_size, alignment)`.

That is why row-major tensors often "look" unaligned logically but are still safe to traverse page-by-page through a tensor accessor: the per-page gap between logical end-of-data and the next page start is part of the buffer contract.

## 3. What Page-Based NOC Helpers Actually Use

Most higher-level kernel-side helpers do **not** use raw `page_size` when an address generator provides an aligned page size. They prefer `get_aligned_page_size()`.

See:

- `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1052`

That is important because a `TensorAccessor` carries the aligned page size as part of its state:

- `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71`
- `tt_metal/hw/inc/api/tensor/tensor_accessor.h:96`

So the generic model is:

1. host/device buffer owns padded page spacing,
2. accessors encode that padded spacing,
3. page-based NOC helpers move one aligned page at a time.

If a kernel bypasses that and calls raw `noc_async_read` / `noc_async_write` with ad hoc addresses and sizes, **the kernel** owns the alignment problem.

## 4. Why Row-Major Inputs Are the Problem Case

For a tiled tensor, page sizes are naturally large and alignment-friendly (a tile is already much wider than any hardware alignment).

For a row-major tensor, one "page" is often just one row/stick:

- `width_in_elements * element_size`

That can be awkward. For BF16:

- 3 channels = 6 bytes
- 7 channels = 14 bytes
- 8 channels = 16 bytes

Small-channel row-major tensors can easily violate the minimum NOC-friendly width unless something earlier in the pipeline pads them or chooses a sharding/channel alignment that fixes it.

A useful mental model for BF16 row-major sticks:

- 8 channels = 16 bytes = minimally L1-safe
- 16 channels = 32 bytes = also safe for Wormhole DRAM reads
- 32 channels = 64 bytes = also safe for Blackhole DRAM reads

## 5. The Three Strategies Available to an Op

When an op encounters a row-major input whose logical row width is smaller than the relevant hardware alignment, there are essentially three options. Any new op has to pick one consciously.

### Strategy A: Reject Unaligned Input

The op validates up front that the input already satisfies alignment, e.g.:

- `page_size == aligned_page_size`, or
- `page_size % alignment == 0`

This is the safest pattern when the kernel assumes page-sized reads/writes directly from CBs or tensor accessors. The cost is paid by the caller (they have to pad, reshape, or shard differently before invoking the op). Many existing ops in the codebase take this route.

### Strategy B: Bounce Through an Aligned Scratch Buffer

The op accepts the unaligned input and the kernel reads it into a temporary aligned buffer (often a small L1 scratchpad sized to one aligned stick), then copies/memmoves into the final destination. The pattern is:

1. compute `aligned_stick_nbytes = align(stick_nbytes, alignment)`,
2. allocate a CB / scratch region sized to the aligned width,
3. issue NOC reads/writes against the aligned width,
4. drop the tail padding when consuming the data.

The cost is an extra copy and the extra L1 footprint of the scratch buffer.

### Strategy C: Pre-Align by Construction

The op (or a separate normalization pass it owns) materializes an aligned row-major representation before the main kernel runs. Concretely this means:

- choose a shard / channel / page width that rounds up to the relevant alignment,
- write a normalization step that produces that aligned layout,
- have the main kernel only ever see the aligned layout.

The cost is the up-front normalization work and the extra memory footprint of the padded layout. The benefit is that the hot path never deals with raw unaligned input.

These strategies are not mutually exclusive within one op family — different sharding modes or different input layouts may pick different strategies.

## 6. Practical Rules for Op Authors

1. Do not reason only about `page_size`. In TT-Metal, device page traversal is usually governed by `aligned_page_size`.

2. If you use `TensorAccessor` or `noc_async_read_page` / `noc_async_write_page`, alignment is normally handled through the accessor's aligned page size — you mostly just have to not fight it.

3. If you use raw `noc_async_read` / `noc_async_write`, you must ensure the addresses, sizes, and per-row stride are safe yourself. The runtime does not retroactively fix unaligned transactions.

4. For row-major sharded tensors, width in **bytes** should usually be aligned to `buffer()->alignment()`.
   - L1-backed row-major shards: usually 16 bytes.
   - DRAM-backed transfers: 32 bytes on Wormhole, 64 bytes on Blackhole.

5. When an op validates `page_size == aligned_page_size`, it is telling you the kernel was written under the assumption that no per-page padding gap exists beyond the natural page size. If you want that op to accept narrower rows, you are picking Strategy B or C, not relaxing the check.

6. Be explicit about logical vs physical row width in your own code. If you carry both `logical_row_bytes` and `physical_row_bytes` (or `stick_nbytes` and `aligned_stick_nbytes`) as separate variables, alignment bugs become easy to see. If you collapse them into one variable, they become invisible.

7. If your kernel writes to a CB or buffer that a later kernel will treat as page-aligned, you are responsible for either honoring that stride (including any tail padding) or documenting that the next stage must repack.

## Source Pointers

- `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h`
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h`
- `tt_metal/impl/allocator/l1_banking_allocator.cpp`
- `tt_metal/impl/buffers/buffer.cpp`
- `tt_metal/hw/inc/api/dataflow/dataflow_api.h`
- `tt_metal/hw/inc/api/tensor/tensor_accessor.h`
