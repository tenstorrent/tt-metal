# Block-tilize Metal 2.0 port: heap corruption in dispatch command construction

**Status:** real bug, NOT yet fixed. The `tilize_multi_core_block` Metal 2.0 port (`create_program_spec`)
crashes with `malloc_consolidate(): invalid chunk size` for **every** block-routed shape, so the block
factory has never actually run at runtime in the port. The other four tilize factories
(single / default / sharded / width-sharded) are fine. Port is committed at `e2bef044147` (branch
`dgomez/rand-metal2`); was NOT reverted to descriptor.

## Where the crash is (gdb backtrace, not a guess)
- `create_program_spec` returns a **clean, valid spec** (passes validation; both internal checkpoints fire).
- Crash is downstream during enqueue:
  `EnqueueMeshWorkload` → `BatchedTransferGenerator::construct_commands`
  → `std::vector<program_dispatch::Transfer>::_M_realloc_insert` → `malloc` trips already-corrupted heap.
- Root cause class: **command-count under-reservation**. `tt_metal/impl/program/dispatch.cpp:~910` has a
  `TT_ASSERT(command_count >= runtime_args_command_sequences.size(), "Incorrect number of commands reserved
  ... Vector reallocation causes cached addresses to be incorrect.")`. `build_Release` is `-DNDEBUG`, so
  this assert is **compiled out** — the vector reallocs silently and corrupts the heap instead of asserting.

## Trigger
Forward tilize of a block-routed shape, e.g. `(32, 15936)` (single tile-row, 498 tiles wide, no cliffs;
`full_cores_per_col=0`) and `(2048, 4096)` (fcpc=4). The spec structure is **2+ WorkUnitSpecs**, each with
`{temp (self-loop PRODUCER+CONSUMER on reader) + src0 + output}` DFBs and reader/writer/compute kernels.
Suspected untested combination: **self-loop DFB across multiple work-units** (rand uses a self-loop but is
single-work-unit and works; matmul is multi-work-unit but has no self-loop).

## Repro
```python
import torch, ttnn
dev = ttnn.open_device(device_id=0)
x = torch.randn((32, 15936), dtype=torch.bfloat16)
t = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=dev)
tile = ttnn.to_layout(t, layout=ttnn.TILE_LAYOUT)   # forward block tilize -> malloc_consolidate abort
```

## Next step to fix (deferred)
1. Build an **assert-enabled** config (re-enable `TT_ASSERT`; cheaper than full ASAN). Rerun repro →
   `dispatch.cpp:910` assert fires cleanly and names the kernel-group whose command count is under-estimated.
2. Fix the command-count estimator in `dispatch.cpp` (Audrey's dispatch code) for this spec shape, OR
   adjust the factory's DFB/work-unit structure if the spec is the thing that's unusual.
3. Validate `tests/.../test_tilize.py::test_run_tilize_large_row_input[(32,15936)]` runs without abort.

This is a metal dispatch-layer issue surfaced by the port, not bad host arithmetic in the factory
(`create_program_spec` is clean).
