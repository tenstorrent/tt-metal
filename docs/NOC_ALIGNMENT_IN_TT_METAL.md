# NOC Alignment in TT-Metal

This note summarizes how NOC alignment works in TT-Metal, why row-major tensors are the awkward case, and how ops like convolutions avoid alignment problems.

## Short Version

There are really two different notions of size:

- `page_size`: the logical amount of tensor data in one page/row/stick.
- `aligned_page_size`: the physical stride used when pages are laid out in device memory and when most page-based NOC helpers step through memory.

The important consequence is:

- a row-major page can be logically narrow,
- but if it lives in a device buffer it is usually **spaced in memory** by an aligned stride,
- and kernels either:
  - operate using that aligned stride, or
  - reject the input, or
  - explicitly bounce through an aligned scratch buffer.

Convolution mainly survives row-major alignment issues by:

1. forcing channel/shard widths to a safe alignment,
2. materializing halo output with an aligned row stride, and
3. having the halo kernels read/write with `aligned_stick_nbytes`, not the raw logical row width.

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

So if your kernel touches DRAM, the DRAM-side alignment matters even when the local buffer only needs 16-byte L1 alignment.

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

That is why row-major tensors often "look" unaligned logically but are still safe to traverse page-by-page through a tensor accessor.

You can also see this assumption in generic row-major kernels. For example, the typecast path treats one row-major page as "one full row including padding", not just the unpadded logical row width:

- `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp:42`

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

If you bypass that and call raw `noc_async_read` / `noc_async_write` with ad hoc addresses and sizes, then **you** own the alignment problem.

## 4. Why Row-Major Inputs Are the Problem Case

For a tiled tensor, page sizes are naturally large and alignment-friendly.

For a row-major tensor, one "page" is often just one row/stick:

- width in elements * element size

That can be awkward:

- BF16 with 3 channels is 6 bytes
- BF16 with 7 channels is 14 bytes
- BF16 with 8 channels is 16 bytes

So small-channel row-major tensors can easily violate the minimum NOC-friendly width unless the op pads them or chooses a sharding/channel alignment that fixes it.

TT-Metal code explicitly codifies this for convolution:

- minimum safe row-major input channel alignment is 8 elements for BF16-like data because L1 alignment is 16 bytes.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:77`

That helper also prefers larger safe widths when possible:

- use 32 if shard width is tile-aligned,
- else 16,
- else 8,
- else fall back to 32.

## 5. How Convolution Gets Around Alignment Problems

The main trick is that convolution does **not** consume arbitrary row-major input rows directly as-is.

### 5.1 Channel Alignment Is Chosen Up Front

`get_input_channels_alignment()` makes the row-major input channel dimension large enough to be safe for NOC movement.

For row-major conv inputs, this is the first line of defense:

- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:77`

### 5.2 The Halo Op Materializes an Aligned Row-Major View

Regular conv goes through the halo path before the conv micro-op:

- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp:252`

The halo/untilize program factory computes:

- `stick_nbytes = output_shard_shape[1] * out_nbytes`
- `aligned_stick_nbytes = round_up(stick_nbytes, input_tensor.buffer()->alignment())`

and for row-major input (`skip_untilize`) it makes the input CB page size equal to that aligned stick width.

See:

- `ttnn/cpp/ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.cpp:173`

This is one of the key escape hatches:

- the logical row may be narrow,
- but the kernel reads it using an aligned stride that matches the actual buffer layout.

### 5.3 The Halo Kernels Use the Aligned Stick Width Everywhere

The halo gather kernel templates are instantiated with `aligned_stick_nbytes`, and even the padding copy path is sized to avoid misaligned NOC transactions.

See:

- `ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp:295`

So by the time the conv micro-op runs, the row-major halo output has already been normalized into an aligned layout that the downstream kernels can move safely.

## 6. Common Escape Hatches Used by Other Row-Major Ops

There are two broad patterns outside conv.

### Pattern A: Reject Unaligned Row-Major Shards

Many ops do not try to be clever. They simply require:

- `page_size == aligned_page_size`, or
- `page_size % alignment == 0`

Examples:

- `all_gather`: `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.cpp:37`
- `tilize` on sharded row-major input: `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_device_operation.cpp:118`
- `pad` on sharded row-major input/output: `ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_device_operation.cpp:184`
- `conv3d` on sharded row-major input: `ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_device_operation.cpp:57`

This is the safest pattern when the kernel assumes page-sized reads/writes directly from CBs or tensor accessors.

### Pattern B: Use an Intermediate Aligned Scratch Buffer

Some row-major data movement ops explicitly repack.

The fold DRAM path is a clean example:

- it computes `aligned_stick_nbytes = align(stick_nbytes, hal::get_dram_alignment())`
- allocates an aligned source CB,
- and if the logical stick width is not aligned, it uses an intermediate L1 scratch buffer.

See:

- `ttnn/cpp/ttnn/operations/data_movement/fold/device/fold_multi_core_dram_program_factory.cpp:292`
- `ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2dram_for_rm_input.cpp:34`

The `pad_multi_core` example shows the same idea more directly:

- read the unaligned row into a temporary aligned buffer,
- then copy/memmove into the final destination.

See:

- `tt_metal/programming_examples/pad_multi_core/kernels/pad_reader_dims_rm_interleaved.cpp:83`

## 7. Important Rules to Remember

If you are writing kernels or new ops, these are the practical rules that matter.

1. Do not reason only about `page_size`.
   In TT-Metal, device page traversal is usually governed by `aligned_page_size`.

2. If you use `TensorAccessor` or `noc_async_read_page` / `noc_async_write_page`, alignment is usually handled through the accessor's aligned page size.

3. If you use raw `noc_async_read` / `noc_async_write`, you must ensure the addresses, sizes, and per-row stride are safe yourself.

4. For row-major sharded tensors, width in **bytes** should usually be aligned to `buffer()->alignment()`.
   For L1-backed row-major shards that usually means 16 bytes.
   For DRAM-backed transfers, the effective constraint can be 32 bytes on Wormhole or 64 bytes on Blackhole.

5. Convolution's row-major path is special because it actively engineers an aligned intermediate representation.
   Do not assume that behavior exists in unrelated ops.

6. When an op validates `page_size == aligned_page_size`, it is telling you the kernel is written under the assumption that no per-page padding gap exists beyond the natural page size.

7. A useful mental model for BF16 row-major sticks:
   - 8 channels = 16 bytes = minimally L1-safe
   - 16 channels = 32 bytes = nicer for Wormhole DRAM reads
   - 32 channels = 64 bytes = nicely aligned even for Blackhole DRAM reads

## 8. My Read on "How Convs Get Away With It"

The answer is not that conv ignores alignment.

The answer is that conv moves the problem earlier in the pipeline:

- choose a safe channel alignment,
- run halo on a row-major representation whose stick stride is explicitly rounded up,
- do all NOC movement using the aligned stick width,
- then let the conv micro-op consume that normalized buffer.

So the row-major conv path is really an **aligned row-major transport path**, not an arbitrary unaligned one.

## 9. What The Conv "Pre-Tilize" CB Actually Contains

This is the main point that matters if you are trying to understand TTNN conv internals:

- the pre-tilize CB is **not tiled yet**,
- it is a **row-major activation matrix** produced by the reader/im2col stage,
- compute then tilizes that matrix into `ACT_TILIZED`,
- matmul consumes the tilized version.

The CB names depend on the sharding mode:

- height-sharded conv:
  - `ACT` is the img2col / pre-tilize CB,
  - `ACT_SECOND_READER` may hold the other half when split reader is enabled,
  - `ACT_TILIZED` is the post-tilize CB.
- width-sharded / block-sharded conv:
  - `ACT_ROW_MAJOR_BFLOAT16` is the row-major pre-tilize CB,
  - `ACT_TILIZED` is the post-tilize CB,
  - `ACT` is often used for multicast / already-tilized activation transport.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.cpp:225`
- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.cpp:256`
- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.cpp:270`

The logical shape of that pre-tilize buffer is:

- rows = output patch positions for the current activation block (`M`)
- cols = flattened receptive-field slice for the current channel block (`K`)

So the reader is effectively materializing an `M x K` matrix for matmul, just not in tile layout yet.

For the height-sharded reader, the code writes row-major segments directly into the CB:

- for dilation 1 it copies one coalesced chunk per selected sliding-window position,
- then advances the output pointer by `coalesced_read_bytes + act_block_w_extra_align_bytes`.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp:61`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp:65`

That `act_block_w_extra_align_bytes` is important. It means each logical im2col row can have **explicit tail padding**. So the pre-tilize CB is row-major, but it is not necessarily tightly packed row-major.

Another important detail is that the reader pushes data in tile-height chunks:

- every 32 produced rows, it does `push_back(act_cb_w_tiles)`.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp:108`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp:148`

So the best mental model is:

- payload state: row-major im2col matrix,
- CB accounting: tile-sized pages,
- production granularity: 32 matrix rows at a time,
- next stage: tilize into standard MM tiles.

### 9.1 Worked Example: BF16 RGB Input (`C = 3`)

This is the easiest way to visualize it.

For row-major conv, TTNN first enforces an input-channel alignment. For BF16 row-major input, the important minimum is 8 channels, because 8 BF16 values = 16 bytes, which is the L1-safe width.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:77`
- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:606`
- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp:152`

So an RGB pixel is not treated as a 3-element stick by the conv reader. It is usually normalized to an 8-element stick:

```text
[r, g, b, 0, 0, 0, 0, 0]
```

for BF16, that is:

- 8 scalars
- 16 bytes

That means the pre-tilize CB is built from padded channel groups.

If the kernel is `1x1`, one im2col row is just one padded pixel:

```text
row_0 = [p0.r, p0.g, p0.b, 0, 0, 0, 0, 0]
row_1 = [p1.r, p1.g, p1.b, 0, 0, 0, 0, 0]
...
```

So the pre-tilize CB for `1x1` / RGB is just a row-major matrix with:

- `M` rows = output pixels,
- `K = 8` columns per channel block.

If the kernel is `3x3`, each output patch row concatenates 9 of those padded sticks in kernel order:

```text
row_i =
[
  p00.r, p00.g, p00.b, 0, 0, 0, 0, 0,
  p01.r, p01.g, p01.b, 0, 0, 0, 0, 0,
  p02.r, p02.g, p02.b, 0, 0, 0, 0, 0,
  p10.r, p10.g, p10.b, 0, 0, 0, 0, 0,
  p11.r, p11.g, p11.b, 0, 0, 0, 0, 0,
  p12.r, p12.g, p12.b, 0, 0, 0, 0, 0,
  p20.r, p20.g, p20.b, 0, 0, 0, 0, 0,
  p21.r, p21.g, p21.b, 0, 0, 0, 0, 0,
  p22.r, p22.g, p22.b, 0, 0, 0, 0, 0
]
```

So logically:

- `K = Kh * Kw * C_aligned = 3 * 3 * 8 = 72`

For the common height-sharded path, that row is then physically widened again to a tile-friendly width:

- `act_block_w_extra_align_bytes` pads each row up to a multiple of 32 scalars.

So for BF16 RGB with `3x3`:

- logical payload per row = 72 BF16 values,
- physical row width in the pre-tilize CB = 96 BF16 slots if this path rounds to the next 32-scalar boundary,
- the last 24 slots are padding, not real activation payload.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp:601`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp:65`

Important nuance:

- spatial halo padding becomes all-zero padded sticks,
- channel padding becomes zero-valued lanes in each stick,
- row-tail padding from `act_block_w_extra_align_bytes` is padding space in the pre-tilize matrix, not meaningful data.

So if you dump the CB for an RGB input, you should not expect to see compact `RGBRGBRGB...`.
You should expect to see:

- `RGB00000` per spatial position,
- repeated across the kernel footprint,
- then possibly additional row-tail padding to make the row tile-friendly.

## 10. How That State Changes In The Optimized Paths

### 10.1 Activation Reuse

Activation reuse does not change the fact that the CB holds a row-major im2col matrix. What changes is **how much new data is written**.

- the first output image width is written as a full window,
- later widths reuse previously written rows,
- only the new kernel-width slice is appended / refreshed.

The pointer math for this is driven by `window_reuse_offset`, and the reader explicitly rewinds / repositions the CB write pointer when moving to the next output image width.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp:89`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp:206`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp:255`

So with activation reuse, the pre-tilize CB is still an `M x K` matrix view, but it is maintained with **row reuse and pointer wraparound**, not rebuilt from scratch every time.

### 10.2 Split Reader

Split reader also does not change the representation. It just splits ownership of the row-major pre-tilize matrix across two CBs:

- `ACT`
- `ACT_SECOND_READER`

Then compute tilizes both into `ACT_TILIZED`.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp:251`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp:357`

### 10.3 Width-Sharded / Block-Sharded

In width-sharded conv, the row-major pre-tilize buffer is very explicit:

- reader fills `ACT_ROW_MAJOR_BFLOAT16`,
- compute tilizes it,
- then a tilized activation block may be multicast to other cores.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp:99`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp:165`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp:202`

That is a useful sanity check: TTNN itself treats "im2col output" and "tilized MM input" as two distinct states.

## 11. Other Big Pieces You Must Solve Besides Im2col

Yes, weights preparation is one of the major other problems. It is not optional.

### 11.1 Weight Layout Conversion

For the simple interleaved-MM path, TTNN converts weights from conv layout:

- `[Co, Ci, Kh, Kw]`

into MM layout:

- `[1, 1, KhKwCi, Co]`

and then tilizes / pads it to tile boundaries.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp:272`
- `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp:288`
- `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp:1569`

That matches the activation-side `M x K` matrix:

- activations are `M x K`,
- weights are `K x N`,
- output is `M x N`.

If you get that contract wrong, the conv may look "almost right" but still be fundamentally broken.

### 11.2 Weight Padding Depends On The Parallelization Scheme

The non-interleaved paths do more than simple tilization:

- input channels are padded to `num_input_channel_cores * input_channels_alignment`,
- output channels are padded to match output shard shape,
- block-sharded and height-sharded paths use specialized weight transforms.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp:1573`
- `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp:1585`

So "prepare weights" really means:

- choose the exact `K` partitioning,
- pad `K` consistently with activation padding/alignment,
- choose a weight tilization/layout that matches the chosen sharding scheme.

### 11.3 Bias Preparation Also Has To Match

Bias is padded to tile width / output shard width, converted to tile layout, and cast to the same dtype as weights.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp:1622`

### 11.4 Halo And Reader Indices Are Part Of The Contract

In the normal conv path, TTNN does not feed the raw NHWC tensor directly to the conv reader. It first runs halo, and the conv micro-op then consumes that halo-normalized row-major tensor.

See:

- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp:252`
- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:941`

The reader then uses precomputed reader indices to walk the sliding windows. If your reimplementation changes the patch ordering, the reader-index generation, weight layout, and output interpretation all have to change together.

## 12. If I Had To Reimplement NHWC Conv In TTNN

I would explicitly separate the job into a **correctness-first** version and a **performance** version.

### Phase 1: Correctness-First

1. Keep the logical model simple:
   - input is NHWC,
   - output patches define `M`,
   - flattened receptive field defines `K = Kh * Kw * C`,
   - output channels define `N`.

2. Materialize the activation-side im2col matrix explicitly as row-major `M x K`.
   - rows should be in the exact order you want the output NHW traversal to use,
   - columns should be in the exact order the weight transform uses: usually `kh`, then `kw`, then `c`.

3. Materialize weights explicitly as `K x N`.
   - start from `[Co, Ci, Kh, Kw]`,
   - reorder to `[KhKwCi, Co]`,
   - pad only after the ordering is proven correct.

4. Make padding/alignment visible in the implementation.
   - keep `logical_row_bytes`,
   - keep `physical_row_bytes`,
   - keep `K_logical` and `K_padded`,
   - do not hide those behind one variable.

5. Only after the row-major `M x K @ K x N` path is correct, add tilize.

### Phase 2: Performance

Then add these one by one:

- halo / sharded halo output,
- channel alignment policy,
- tilize into `ACT_TILIZED`,
- split reader,
- activation reuse,
- multicast,
- output untilize / row-major output path,
- stride folding.

If you add those too early, you lose the ability to tell whether a wrong answer comes from:

- patch ordering,
- halo math,
- weight layout,
- padding/alignment,
- or reuse/split-reader pointer bugs.

## 13. The Main Potholes

These are the issues I would warn someone about first.

1. Do not confuse logical im2col shape with physical row stride.
   `act_block_w_extra_align_bytes` means the pre-tilize CB can be row-major but still padded per row.

2. Keep the activation `K` ordering and weight `K` ordering identical.
   The activation reader and `prepare_conv2d_weights.cpp` must agree on the exact flattening order.

3. Do not treat halo as incidental.
   In TTNN conv it is part of the input normalization contract, not just a convenience wrapper.

4. Channel alignment is not just a performance knob.
   For row-major paths it is part of making the NOC transfers legal at all; the important floor is 8 BF16 channels = 16 bytes.

5. Activation reuse is easy to get subtly wrong.
   `window_reuse_offset`, image-width padding to tile width, and batch transitions all affect pointer movement.

6. Split reader is representation-preserving but pointer-sensitive.
   If the two readers do not agree on where their halves land, tilize will silently consume the wrong rows.

7. Width-sharded, block-sharded, and height-sharded convs are not the same algorithm with a different launch grid.
   They use different CB roles and different activation/weight transport strategies.

8. Kernel-stride folding changes both the input interpretation and the weight interpretation.
   If you support it, treat it as a separate transform layer, not a small tweak.

9. Bias/output handling matters.
   Bias padding, fused bias, and row-major output/untilize are separate contracts that can break correctness even when the main matmul is right.

10. Build a reference view of the pre-tilize CB.
   If you cannot dump and explain one activation block as an `M x K` matrix with row padding, you are debugging blind.

## Source Pointers

- `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h`
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h`
- `tt_metal/impl/allocator/l1_banking_allocator.cpp`
- `tt_metal/impl/buffers/buffer.cpp`
- `tt_metal/hw/inc/api/dataflow/dataflow_api.h`
- `tt_metal/hw/inc/api/tensor/tensor_accessor.h`
- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp`
- `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp`
- `ttnn/cpp/ttnn/operations/data_movement/fold/device/fold_multi_core_dram_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2dram_for_rm_input.cpp`
