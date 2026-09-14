# A borrowed buffer's per core size is checked against the whole tensor size

## The request

Pull request #56494 converts one operation, `data_movement/slice`, to a newer
host API called Metal 2.0. A host API is the set of C++ functions that run on the
CPU to describe and launch work on the accelerator.

After the conversion, two sharded slice cases that work correctly today abort with
a failed assertion inside the framework, before any work reaches the device.

The conversion did not change either buffer's size. The assertion compares a per
core size against a whole tensor size, which are not the same kind of quantity.
The fix belongs in the framework, and **this pull request should wait for it**
rather than carry it.

## Background

### SRAM, and how a tensor is spread over it

A Tenstorrent accelerator holds many cores. Each core has its own small, fast
memory called SRAM. A tensor placed in SRAM does not sit on one core: it is cut
into pieces, one per core. The piece on a single core is a **shard**, and the
arrangement is **sharding**. The opposite placement is **interleaved**, where the
tensor is spread across memory banks rather than held one piece per core.

This document needs only **height sharding**: the tensor is cut into groups of
whole rows, and each core gets one group.

### Rows, unpadded row size, and stride

A tensor in **row-major** layout is stored one row after another. The number of
bytes of real data in a row is its **unpadded row size**.

Each row must start at an address that is a multiple of an alignment value. On
this device that value is 16 bytes; the framework carries it as a parameter, not
a constant. A row whose unpadded size is not a multiple of 16 is followed by
unused padding. The distance from the start of one row to the start of the next is
the **stride**.

Take a tensor of 32 rows of 52 values in `bfloat16`, so 2 bytes per value:

- unpadded row size = 52 x 2 = **104 bytes**
- stride = 104 rounded up to a multiple of 16 = **112 bytes**

Each row occupies 112 bytes, of which 8 are padding.

### Where a shard height comes from

A shard's height is part of the tensor's memory configuration, chosen by the
caller or by a helper that divides rows over cores and rounds up. It is therefore
a separate number from the tensor's row count, and it can exceed it: an 18 row
tensor can sit in a shard sized for 32 rows, leaving 14 reserved rows that hold
nothing. In the code the shard height is `shard_height_padded`. The "padded" there
refers to whole unused rows, not to the per row padding above.

### Two sizes for one sharded tensor, measured on different things

**Packed size** counts only real data across the whole tensor, with no padding and
no unused rows:

```
rows in the tensor  x  unpadded row size
```

**Allocated size per core** counts the memory reserved on one core:

```
rows in the shard  x  stride
```

Packed size is a property of the whole tensor. Allocated size per core is a
property of one core. Two independent effects make the per core figure larger than
rows in shard x unpadded row size: padding within each row, and whole rows a shard
reserves but never uses.

For the 32 x 52 tensor on one core, with a 32 row shard:

- packed size = 32 x 104 = **3328 bytes**
- allocated size per core = 32 x 112 = **3584 bytes**

The kernel addresses whole shard rows at stride spacing, so the region it can
address is rows in shard x stride, 3584 bytes.

### Circular buffers, dataflow buffers, and borrowing

A kernel needs a named region of SRAM to read and write. In the older host API
that region is a **circular buffer**; in Metal 2.0 it is a **dataflow buffer**,
abbreviated `DFB` in code and in error messages. For this document they are the
same thing under two names.

Normally the framework allocates fresh SRAM for such a buffer. But when the data
is already in SRAM, because the tensor is sharded there, a second copy would waste
memory, so the buffer is pointed at the memory the tensor already occupies. Both
APIs call this **borrowing**.

A borrowed buffer is described by two numbers. One entry holds one row, so
`entry_size` is the stride, and `num_entries` is the shard height. Their product
is the size of the region the kernel will address. The older API calls one entry a
**page**, so its `page_size` is the same stride that Metal 2.0's `entry_size`
holds; the alignment value above is a page alignment in both APIs.

### Program factories

A **program factory** is the code that builds the device program for one class of
inputs. An operation has several and picks one from the layout of its tensors.
Slice has a factory for height-sharded row-major tensors, which is the one this
document concerns.

## The conversion did not change the sizes

Before the conversion, at the pull request's base commit `a5d0ee0d5`, the factory
declared two borrowed circular buffers. Lines 291-300 and 303-312 of
`ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.cpp`
at that commit read, for the input:

```cpp
.total_size = shard_height_padded * src_stride_bytes,
.page_size  = src_stride_bytes,
.buffer     = input.buffer(),
```

After the conversion it declares two borrowed dataflow buffers, at
[slice_program_factory_rm_sharded.cpp:296-309](ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.cpp#L296-L309):

```cpp
.entry_size    = src_stride_bytes,
.num_entries   = shard_height_padded,
.borrowed_from = INPUT,
```

Both give the same product of the same two values. That total is the allocated
size per core, 3584 bytes in the example.

## The defect

`TT_FATAL` aborts the program with a message when its condition is false.

Both APIs assert that a borrowed buffer is no larger than the memory it borrows.
They differ in what they compare against.

**Metal 2.0's early check, when the program is described.** No buffer object
exists yet, only a description of the tensor, held in a type called `TensorSpec`.
This check uses the packed size, at
[program_spec.cpp:1618-1620](tt_metal/impl/metal2_host_api/program_spec.cpp#L1618-L1620):

```cpp
const size_t tensor_bytes = tensor_spec.compute_packed_buffer_size_bytes();
TT_FATAL(dfb_bytes <= tensor_bytes, ...);
```

`dfb_bytes` is the buffer's `entry_size` x `num_entries`, a per core figure.
`tensor_bytes` is the packed size of the whole tensor. The two sides are not the
same kind of quantity, and nothing scales one to the other. The absence of a
buffer object does not force this choice: as shown below, the allocated size per
core is also computable from the description alone.

**Metal 2.0's late check, when the program runs.** The tensor's buffer object
exists by then, so this check uses the allocated size per core, at
[program_run_args.cpp:535-537](tt_metal/impl/metal2_host_api/program_run_args.cpp#L535-L537).
A buffer covering one whole shard passes.

**The older API** read the allocated size per core from the tensor's buffer object
and stored it as the limit at
[circular_buffer_config.cpp:226](tt_metal/impl/buffers/circular_buffer_config.cpp#L226),
then enforced it at
[circular_buffer_config.cpp:196](tt_metal/impl/buffers/circular_buffer_config.cpp#L196).
A whole-shard buffer passes, because it equals the limit.

So a buffer that correctly covers one whole shard is accepted by the older API and
by Metal 2.0's late check, and rejected by Metal 2.0's early check. The early
check fails when:

```
rows in shard x stride  >  rows in tensor x unpadded row size
```

The core count decides whether this can hold. With N cores and shards exactly
filled, the right side is N x rows in shard x unpadded row size, so the check can
fail only when stride exceeds N x unpadded row size. Since 16 byte alignment adds
at most 15 bytes, that needs an unpadded row size under 16 bytes once N is 2 or
more. One core is where the check can fail on ordinary sizes.

The comment above the early check mentions only the reverse direction, at
[program_spec.cpp:1612-1615](tt_metal/impl/metal2_host_api/program_spec.cpp#L1612-L1615):
"a DFB can pass here against the full-tensor size and still fail per-bank later.
By design." Passing late and failing early is not mentioned.

### Observed failures

Two separate single-core height-sharded row-major slice calls, both `bfloat16`.
Each aborts on the first call. The number after "shard" is the shard height in
rows, so "32 x 52 shard 32" is a 32 row, 52 column tensor in a shard 32 rows tall.
The four numeric columns describe the reported buffer's tensor:

| Failing case, input to output | Buffer reported | Unpadded row size | Stride | Buffer bytes | Packed bytes |
|---|---|---|---|---|---|
| 32 x 52 shard 32, to 26 x 52 shard 26 | input | 104 | 112 | 32 x 112 = 3584 | 32 x 104 = 3328 |
| 32 x 64 shard 32, to 18 x 64 shard 32 | output | 128 | 128 | 32 x 128 = 4096 | 18 x 128 = 2304 |

The first is rejected because of per row padding, the second because of unused
shard rows. The buffers are checked in order, input first, so only the first one
over the packed size of its tensor is reported. In the first case the output buffer
is over it too, at 26 x 112 = 2912 against 26 x 104 = 2704. In the second case the
input buffer is exactly at the packed size, 4096 against 4096, and passes.

The messages name these numbers:

```
DFB 'sharded_in'  (entry_size 112 * num_entries 32 = 3584 bytes) is larger than
  its borrowed TensorParameter 'input'  (3328 bytes).
DFB 'sharded_out' (entry_size 128 * num_entries 32 = 4096 bytes) is larger than
  its borrowed TensorParameter 'output' (2304 bytes).
```

Three control cases pass and match an unsharded reference bit for bit: one core
with width 64 and exactly filled shards; two cores with width 64; and two cores
with width 52, which has the same per row padding as the first failing case but
passes because the packed size counts both shards. So the failure is specific to a
buffer whose per core size exceeds the whole tensor packed size, not to sharded
slice in general.

No existing test catches this. The one single-core sharded slice test uses tiled
layout, which selects a different program factory.

### The same check rejects an operation this pull request does not touch

`sharded_to_interleaved` was converted to Metal 2.0 earlier, so it already meets
the early check. It sizes its borrowed buffer the same way, stride times shard
rows, at
[sharded_to_interleaved_program_factory.cpp:113-117](ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp#L113-L117).
On the pull request's base commit, an 18 row `bfloat16` tensor under a one core 32
row shard fails the same early check with the same message.

That operation is untouched by this pull request, which places the cause in the
check rather than in the slice conversion.

## The proposed framework change

For a borrowed buffer whose tensor lives in SRAM, the early check should compare
against the allocated size per core: the quantity the late check already uses, and
the quantity the kernel addresses.

That figure is computable from the tensor description alone, so the early check
can still run at description time. `TensorSpec` already provides it, at
[tensor_spec.hpp:94](tt_metal/api/tt-metalium/tensor/spec/tensor_spec.hpp#L94):

```cpp
size_t compute_consumed_memory_bytes_per_bank(size_t page_alignment, size_t num_banks) const;
```

A bank here is one core's SRAM, so the per bank figure is the allocated size per
core. The caller supplies the alignment and the bank count. This makes the two
checks agree, and loosens the early one only to what the late one enforces anyway.
I have not written the change.

## Why this pull request must not carry the change

### 1. The file is outside what this conversion may touch

This effort limits each conversion to the operation's own directory. The early
check lives in `tt_metal/impl/`, framework code shared by every operation.
Changing it from inside a single-operation conversion puts a change affecting all
operations into a pull request reviewers are reading as a conversion of one.

### 2. It would blur which change caused what

The conversion is meant to preserve behavior, so that any behavior change observed
after it is known to be a defect in the conversion or in the new API. If the same
pull request also relaxes a framework check, that no longer holds: a later problem
in any operation using borrowed buffers could come from either change.

### 3. The repair inside the operation would change kernel logic

One suggestion is to declare a smaller buffer, as `pad`'s height-sharded
row-major factory does at
[pad_rm_sharded_height_only_program_factory.cpp:287](ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_rm_sharded_height_only_program_factory.cpp#L287),
where the entry count is limited to the rows actually used.

That needs a kernel change here. Before writing, the kernel asks the framework for
space for a given number of entries and waits until that much is free; after
writing it hands those entries on. Slice's kernel asks for, and hands on, a count
equal to the shard height, at
[slice_reader_unary_unpad_dims_rm_sharded.cpp:41](ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L41)
and
[line 90 of the same file](ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L90).
Asking for more entries than the buffer holds does not fail; it waits forever for
space that will never exist. Shrinking the buffer therefore means changing those
two calls, which changes what the kernel does rather than how it is described.

### 4. No size the operation could declare is both accepted and correct

The kernel addresses whole shard rows at stride spacing. For the 32 x 52 case that
is 32 x 112 = 3584 bytes. The early check permits at most 3328, and the declared
size is `entry_size` x `num_entries`, so with a 112 byte entry the largest
accepted value is 29 x 112 = 3248 bytes.

- Declare 3584 and the early check rejects it.
- Declare 3248 or less and the description is accepted while the kernel reads and
  writes past the declared end of the buffer.

Declaring 3248 is worse than declaring 3584. The memory is physically present, so
nothing crashes, and the late check passes because it measures against the larger
per core size. The result is a program whose description does not match what it
does, adopted to satisfy a check.

The second failing case is the same. The kernel asks for 32 entries, 4096 bytes;
the check permits 2304 bytes, which is 18 entries.

So the operation cannot describe these cases correctly while the early check
stands as it is.

## Summary

- Metal 2.0's early check compares a per core buffer size against a whole tensor
  packed size. The late check and the older API both use the per core size, and
  both accept these buffers.
- The conversion left both buffers at exactly their previous sizes, so it is not
  the cause. The same check rejects `sharded_to_interleaved`, which this pull
  request does not touch.
- The early check should use `compute_consumed_memory_bytes_per_bank`. That change
  is framework-wide and no size the operation could declare instead is both
  accepted and correct, so it belongs in its own change and this pull request
  should wait for it.
