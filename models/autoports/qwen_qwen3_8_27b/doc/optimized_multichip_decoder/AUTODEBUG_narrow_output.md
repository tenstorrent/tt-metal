# AutoDebug: narrow DRAM matmul output and borrowed-DFB validation

## Verdict

**The native matmul writer stays within the allocated L1 shards for N32 and
N4160.** The N32 failure is a false rejection in the subsequent
sharded-to-interleaved operation's ProgramSpec validation: it compares a
borrowed DFB with the tensor's packed data size, although the backing buffer
allocates a larger full shard.

The allocation-overrun hypothesis is refuted by the allocation and writeback
source chain below. A separate validator defect is source-proven. An isolated
patch corrects that validator and adds host/mock and N32 projection tests;
it is not applied to the root native source and has not been built or run.

The parent reports that padding local N and slicing after conversion passes
strict PCC and replay for both split-projection controls. Those adapted
candidates lose to the selected packed implementation, whose projections
never use N32. This optimization branch is closed by the measured pad/slice
adaptation and comparison. The broader optional API validator patch is
excluded from this stage's implementation scope. The existing coordinate/tail
repairs do not need it for the selected packed outputs.

## Failure evidence

`family_split_attention_l0.log:25` reports:

```text
DFB 'in' (entry_size 2048 * num_entries 2 = 4096 bytes) is larger
than its borrowed TensorParameter 'input' (2048 bytes).
```

The stack names `ShardedToInterleavedDeviceOperation` and
`ValidateProgramSpec`, after the matmul returned. The local projection is
K5120/N32, ten activation/storage-policy cores, two readers per DRAM bank.
The parent has validated and measured a model adaptation that pads local N
and slices after conversion. This report does not infer a kernel overrun from the
validator's misleading packed-size comparison.

## Allocation safety proof

1. The DRAM matmul output-spec branch computes output-core count as
   ceil(Ntiles / per_core_N), with shard shape
   `[per_core_M * tile_height, per_core_N * tile_width]`
   (`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2510-2554`).
   Logical output shape remains unchanged; an oversized final shard is valid.

2. `TensorLayoutImpl::compute_packed_buffer_size_bytes` multiplies physical
   tensor pages by page bytes, without rounding the whole tensor to the
   shard extent (`tt_metal/impl/tensor/spec/layout/tensor_layout.cpp:257-275`).
   It is therefore a packed-data count, not an L1 allocation-capacity count.
   `tensor_impl::allocate_device_buffer` supplies this packed size together
   with the complete sharding arguments (`tt_metal/impl/tensor/tensor_impl.cpp:23-38`).

3. For sharded buffers, `Buffer::num_dev_pages()` uses
   `shard_spec.num_pages() * num_cores`, not the packed page count.
   `ShardSpecBuffer::num_pages()` is the full shard height/width in pages
   (`tt_metal/impl/buffers/buffer.cpp:715-722,827-837`).
   `aligned_size()` and `aligned_size_per_bank()` use that full device-page
   count and aligned page size (same file, lines 763-773).

4. The ordinary allocator passes `buffer->aligned_size()` and
   `buffer->num_cores()` to its L1 bank manager
   (`tt_metal/impl/allocator/allocator.cpp:136-144,178-200`).
   The bank manager substitutes the shard count for the bank count and
   allocates the resulting full per-bank size
   (`tt_metal/impl/allocator/bank_manager.cpp:410-438`). Thus the last
   allocated core gets a full shard, including its padding.

5. The matmul writer's output-core count is ceil(Ntiles / storage_width).
   Its one/two-shard branch splits each reader's output at storage-shard
   boundaries and drops any segment beyond that output-core count. The
   installed tail fix gives exhausted-storage readers zero writes only after
   proving they own no logical columns. The output-column total cannot exceed
   `num_output_cores * storage_width`; that capacity corresponds to actual
   allocated full shards, rather than merely an assumed padded shape
   (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:763-943`).

The relevant concrete BF16 sizes are:

| Quantity | N32, two readers | N4160, two readers |
| --- | ---: | ---: |
| Logical output tiles | 1 | 130 |
| Storage width per core, tiles | 2 | 15 |
| Actual allocated output cores | 1 | 9 |
| Packed tensor bytes | 2,048 | 266,240 |
| Actual capacity per core, bytes | 4,096 | 30,720 |
| Total allocated shard capacity, bytes | 4,096 | 276,480 |
| Matmul output columns written, tiles | 2 | 135 |

For N32, all-padding DRAM banks are removed; one bank's two readers remain.
Each reader produces one tile. Reader 0 writes the logical tile at offset 0.
Reader 1 writes the padding tile at offset 2048 in the same output core.
That write ends at 4096, exactly the end of the allocated two-tile shard.
It does not write beyond an alleged 2048-byte allocation: 2048 is only the
packed size.

For N4160, output capacity is nine full 15-tile shards. Reader 14 writes
columns [126,135), ending at the final shard boundary. Reader 15 owns only
columns [135,144) and writes zero after the installed tail repair. The five
written padding tiles beyond logical column 129 are allocated. With three
readers, reader 22 writes only [132,135), and reader 23 writes zero.

The proof is for the stated B1/tiled BF16 cases and shared shard geometry.
It is not a blanket certification of unrelated matmul variants, arbitrary
tile formats, or heterogeneous allocator geometry.

## Why conversion rejects a valid allocation

`sharded_to_interleaved_program_factory.cpp:55-59,109-118` borrows its input
DFB with the full shard's tile count. For N32 this is two 2048-byte entries.
Its writer separately clips output to the logical tile extents
(same file, lines 248-270), so retaining the full input shard does not make
conversion write two tiles into the one-tile destination.

`tt_metal/impl/metal2_host_api/program_spec.cpp:1600-1618` currently uses
`compute_packed_buffer_size_bytes()` for the spec-time size bound. It
therefore rejects 4096 > 2048 before the valid buffer is attached.

The later `AttachBorrowedDFBBuffers` check already compares against the
correct `buffer->aligned_size_per_bank()`
(`tt_metal/impl/metal2_host_api/program_run_args.cpp:460-480`). It would
accept this shard's 4096-byte borrow. Conversely, the old packed-size check
can incorrectly allow a DFB that fits across several shards but exceeds any
single core; the attach-time check must then reject it.

## Focused capacity observation and ownership incident

The standalone C++ artifact `narrow_output_capacity_probe.cpp` constructed
the two TensorSpecs and queried packed bytes, shard pages, and
`compute_consumed_memory_bytes_per_bank(16, 1)`. It was compiled with the
existing native compiler/includes; its command is preserved in
`narrow_output_capacity_probe.build_command`. Assertions were enabled.
It produced:

```text
N=32 packed=2048 shard_pages=2 allocated_per_core=4096 allocated_total=4096
N=4160 packed=266240 shard_pages=15 allocated_per_core=30720 allocated_total=276480
```

These are capacity calculations from native TensorSpec APIs, not measurements
of live output buffers. The probe called no mesh creation, kernels, reset,
or model operation. However, TensorSpec construction implicitly initialized
MetalContext/UMD. The intended host-only probe unexpectedly opened the driver;
it must not be described as hardware-free:

- Driver opened: 2026-09-12 02:19:46.648 UTC.
- Device start: 02:19:46.891–02:19:47.065 UTC.
- Driver/device close began: 02:19:47.210/211 UTC.
- Device close and cluster destruction completed: 02:19:47.811 UTC.
- Process exited normally with code 0.

This was reported immediately to the parent; no further linked probes or
imports were run. The parent checked timing ownership and reports no overlap:
the preceding failed prefill run had completed driver close at 02:18:18.703,
and the next run began after this probe's close. The source now explicitly
warns about this implicit initialization. Future host checks must configure
the existing mock environment before any TensorSpec construction.

## Isolated patch and validation scope

`narrow_output_validator.patch.gz` changes only:

- `tt_metal/impl/metal2_host_api/program_spec.cpp`: for sharded L1 tensor
  parameters, bound the borrowed DFB against
  `compute_consumed_memory_bytes_per_bank(hal::get_l1_alignment(), 1)`.
  The bank count is unused in the sharded branch. This API handles full shard
  pages and per-page L1 alignment. Interleaved parameters retain their packed
  size check, and the actual attach-time capacity check remains intact.
- `tests/tt_metal/tt_metal/api/metal2_host_api/test_program_spec.cpp`: mock
  tests accept a two-tile DFB on a one-logical-tile/two-tile shard, reject
  three tiles on that shard, and reject a three-tile borrow from a two-core
  tensor whose total packed size is four tiles but whose per-core capacity
  is two tiles.

`narrow_output_projection_test.patch.gz` separately adds K5120/N32/ten storage
cores to the existing reader1/2/3, full/offset-mesh regression. The existing
test already includes native matmul followed by sharded-to-interleaved
conversion, per-rank correctness, fresh-allocation cache hits, and descriptor
placement/writeback assertions.

Both patches are isolated in `/home/mvasiljevic/qwen38-full-rerun/dram-mesh-fix`
and pass `git diff --check` and root `git apply --check`. The Python
addition passes Black and `py_compile`. The validator change and new C++
mock tests are uncompiled/unexecuted and require the normal native build if
accepted. No root model/native/test implementation was edited for this audit.

Proposed checks after deliberate integration:

```bash
# With Metal API tests enabled in the build:
cmake --build build_Release --target unit_tests_api --parallel 4
build_Release/test/tt_metal/unit_tests_api --gtest_filter='ProgramSpecTestQuasar.CPU_BorrowedMemoryDFB*'
python_env/bin/python -m pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py -k single_tile_projection -q
```

The API test target's binary output directory should be confirmed from the
chosen build graph before execution. The projection regression requires
parent-owned hardware serialization and the rebuilt/installed runtime.
