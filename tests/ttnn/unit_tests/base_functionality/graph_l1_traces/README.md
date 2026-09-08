# Reading a ttnn graph-capture trace for L1 accounting

Captured traces for four ops, plus what every node in them means. Companion to
`../test_graph_l1_breakdown.py`, which re-implements the peak accounting in Python and asserts
it against the C++ extractor.

| file | op | mode | nodes |
|---|---|---|---|
| `matmul_l1_out.json` | `ttnn.matmul`, 1024x1024 bf16 DRAM in, L1 out | `NO_DISPATCH` | 22 |
| `matmul_normal_mode.json` | the same matmul | `NORMAL` | 24 |
| `sharded_add.json` | `ttnn.add`, height-sharded L1 over 64 cores | `NO_DISPATCH` | 23 |
| `sharded_add_in_place.json` | the same add with `output_tensor=a` | `NO_DISPATCH` | 16 |
| `conv2d.json` | `ttnn.conv2d` 32->64, 3x3, 128x128, height-sharded | `NO_DISPATCH` | 186 |

All captured on Blackhole p150b. `conv2d.json` came from a build where conv2d/halo still exist.

See `L1_PRESSURE_PROFILING.md` for the consolidated findings and the design of a per-case L1
pressure profiler built on top of these traces, and
`L1_profiling_critical_findings.md` for the program-cache pitfall that invalidates the obvious
`RunMode.NORMAL` measurement design. `PROGRAM_CACHE_BENCHMARK.md` measures the cost of disabling
the program cache across 50 distinct programs with cold and warm persistent kernel caches.
`GRADED_RUN_L1_OVERHEAD.md` measures the incremental cost of cache-off capture and peak reduction
inside the real graded RMSNorm pass. `PERF_PROFILER_OVERHEAD.md` measures the existing eval
device-performance profiler and its interaction with L1 collection.

Tools in this directory:

```bash
# regenerate the JSONs (needs a device)
scripts/run_safe_pytest.sh tests/ttnn/unit_tests/base_functionality/graph_l1_traces/dump_traces.py

# render the views used below (no device)
python3 inspect_trace.py conv2d.json                     # census + skeleton + L1 nodes
python3 inspect_trace.py conv2d.json --view skeleton
python3 inspect_trace.py matmul_l1_out.json --view raw    # whole trace, reflection strings elided
```

---

## 1. What capture is

`ttnn.graph.begin_graph_capture()` pushes a `GraphProcessor` (an `IGraphProcessor`) onto the
process-wide `GraphTracker`. ttnn and metal then call the tracker at fixed points and the
processor appends one **node** to a flat `std::vector<Vertex>`. `end_graph_capture()` pops the
processor and dumps that vector as JSON.

The hook sites that move L1:

| call site | tracker call | node emitted |
|---|---|---|
| `Buffer::allocate_impl` (`tt_metal/impl/buffers/buffer.cpp:582`) | `track_allocate` | `buffer_allocate` |
| `Buffer::deallocate` | `track_deallocate` | `buffer_deallocate` |
| `track_workload` (`ttnn/api/ttnn/mesh_device_operation_utils.hpp:81`), once per program in the workload, right before it is enqueued | `track_program` | `circular_buffer_deallocate_all`, then one `circular_buffer_allocate` per CB |
| `track_program_l1` (`tt_metal/impl/graph/graph_tracking.cpp:113`) | `track_allocate_dataflow_buffer` / `..._scratchpad` | `dataflow_buffer_allocate`, `scratchpad_allocate` |
| the C++ op decorator, on entry/exit | `track_function_start` / `track_function_end` | `function_start`, `function_end` |

`track_program` is also the answer to "I have the `Program` object, how much L1 is that?" — it is
literally:

```cpp
// ttnn/core/graph/graph_processor.cpp:415
for (const auto& cb : program->circular_buffers())
    track_allocate_cb(cb->core_ranges(), 0, cb->size(), cb->globally_allocated(), device);
```

### The two run modes

`ProcessorHooks` (`ttnn/core/graph/graph_processor.cpp:956`) is what separates them:

- **`NORMAL`** — ops really execute, addresses are real.
- **`NO_DISPATCH`** — `hook_allocate` and `hook_program` return true, so `Buffer::allocate_impl`
  sets `address_ = 0` without touching the allocator and programs are never enqueued.

**Sizes are exact in both modes.** They come from the `TensorSpec` and `CircularBufferConfig`, not
from the allocator. Only addresses are fake. That is why `NO_DISPATCH` can price an op that would
not fit.

---

## 2. The node schema

Every node is `{counter, node_type, params, arguments, connections, input_tensors, stacking_level}`,
plus `duration_ns` on `function_end` / `capture_end`.

- `counter` equals the list index and is strictly increasing — the C++ extractor `TT_THROW`s
  otherwise, so **the list is the execution order**.
- `connections` make it a graph. For memory accounting you only need the linear order.
- `stacking_level` is the op nesting depth. This is what separates a composite op from its children.
- `params` values are **strings**, except a whitelist coerced to ints: `size`, `address`,
  `tensor_id`, `device_id`, `buffer_type`, `page_size`, `num_cores`, `max_size_per_bank`
  (`integer_params` in `ttnn/core/graph/graph_processor.cpp:50`). So `globally_allocated` and
  `borrows_memory` arrive as `"0"` / `"1"` — easy to get wrong.
- `params` is `null` on `capture_start` / `capture_end`.

Params by node type:

| node_type | params |
|---|---|
| `function_start` | `name`, `inputs` (a **count**, not the inputs) |
| `function_end` | `name` (+ `duration_ns` on the node) |
| `tensor` | `tensor_id`, `shape`, `dtype`, `layout`, `size` (logical volume x element size), `memory_config`, `address`, `buffer_type`, `device_id`, `device_tensors` |
| `buffer` | identity node, de-duped by `Buffer::unique_id()`: `size`, `type`, `buffer_type`, `layout`, `device_id` |
| `buffer_allocate` / `buffer_deallocate` | `size`, `max_size_per_bank`, `address`, `type` (`DRAM`/`L1`), `buffer_type`, `layout` (`INTERLEAVED`/`HEIGHT_SHARDED`/`WIDTH_SHARDED`/`BLOCK_SHARDED`), `page_size`, `num_cores` (0 = interleaved), `device_id` |
| `circular_buffer_allocate` | `size`, `address`, `core_range_set`, `globally_allocated`, `device_id` |
| `dataflow_buffer_allocate` | `size`, `address`, `core_range_set`, `borrows_memory`, `device_id` |
| `scratchpad_allocate` | `size`, `address`, `core_range_set`, `device_id` |
| `circular_buffer_deallocate_all` | `device_id` |

**Which number is which.** On a buffer node, `size` is the whole buffer across all banks and
`max_size_per_bank` is the per-core slice — the per-core slice is what the peak accounting uses.
On a CB / dataflow-buffer / scratchpad node, `size` is already per core.

`core_range_set` renders as `CoreRangeSet::str()`: `{[x-y - x-y], [x-y - x-y]}`.

`buffer_type` is the `BufferType` enum: `0 DRAM, 1 L1, 2 SYSTEM_MEMORY, 3 L1_SMALL, 4 TRACE`.

---

## 3. The accounting rules

`ttnn.graph.extract_resource_usage_per_core` (`ttnn/core/graph/graph_trace_utils.cpp:308`) walks the
same list, keeps four running totals plus their sum, and reports each high-water mark:
`peak_cb`, `peak_l1`, `peak_dataflow_buffer`, `peak_scratchpad`, `peak_total`.

Two classes of allocation are deliberately **not** added:

- a CB with `globally_allocated == "1"` — a **tensor-backed** CB, i.e. the CB *is* a tensor's L1
  shard (`CircularBufferConfig::set_globally_allocated_address`). Those bytes are already counted
  as the tensor under `peak_l1`.
- a dataflow buffer with `borrows_memory == "1"` — same argument, Metal 2.0 flavour.

`circular_buffer_deallocate_all` zeroes the CB + DFB + scratchpad totals: program-scope L1 lives
exactly as long as the program.

Two behaviours that surprise people:

1. **The capture is a window.** A tensor allocated before `begin_graph_capture` has no
   `buffer_allocate` node, so it contributes nothing to `peak_l1` even though it physically
   occupies L1 the whole time. To price an op honestly, allocate its inputs inside the capture.
2. **`peak_cb` ignores `core_range_set`.** Two CBs on disjoint core sets are summed as if they
   shared a core. That is a deliberate worst case. `Breakdown.peak_per_core` in the companion test
   does the core-aware aggregation.

---

## 4. `matmul_l1_out.json` — the plain case

22 nodes, the whole shape of a trace. Reading it in order:

- **0-1** `capture_start`, then the `ttnn.matmul` frame at `stacking_level: 1`.
- **2-6** `MatmulDeviceOperation` opens at level 2 with `input_tensors: [3, 5]` — pointers to the
  `tensor` nodes it was handed. Both resolve to the **same** `buffer` node 4
  (`"connections": [3, 5]`) because the same tensor was passed twice; `add_buffer` de-dups by
  `Buffer::unique_id()`. There is no `buffer_allocate` for them — allocated before the window.
- **7-10** the output: `buffer` (identity) -> `buffer_allocate` (the event) -> `tensor`.

```json
{ "counter": 8, "node_type": "buffer_allocate",
  "params": { "address": 0, "buffer_type": 1, "device_id": 1, "layout": "INTERLEAVED",
              "max_size_per_bank": 20480, "num_cores": 0, "page_size": 2048,
              "size": 2097152, "type": "L1" },
  "connections": [7], "stacking_level": 3 }
```

  `size` 2 MB whole, `max_size_per_bank` 20480 = the 20 KB/core that counts, `num_cores: 0`
  marks it interleaved, `address: 0` because `NO_DISPATCH` hooked the allocator.

- **11** `circular_buffer_deallocate_all` — `track_program` fires this *before* the program's own
  CBs, releasing whatever the previous program held. Always the first program-scope node.
- **12-14** the three CBs, all `globally_allocated: "0"`, 131072 + 131072 + 524288 = **768 KB per
  core**, on a 4-core range:

```json
{ "counter": 14, "node_type": "circular_buffer_allocate",
  "params": { "address": 0, "core_range_set": "{[0-0 - 1-1]}", "device_id": 1,
              "globally_allocated": "0", "size": 524288 },
  "connections": [], "stacking_level": 2 }
```

- **15-17** `function_end` carries `duration_ns` and re-emits the output tensor node.
- **18-21** the input tensor going out of Python scope: a `buffer_deallocate` *outside* the op.

Totals: `peak_cb` 768 KB, `peak_l1` 20 KB, `peak_total` 788 KB.

---

## 5. `matmul_normal_mode.json` — the same op, real addresses

`python3 inspect_trace.py matmul_normal_mode.json --view l1`, next to the `NO_DISPATCH` run:

```
--- NO_DISPATCH ---
   8 buffer_allocate           {"address": 0,          "max_size_per_bank": 20480, "type": "L1",   "device_id": 1}
  12 circular_buffer_allocate  {"address": 0,          "size": 131072,             "device_id": 1, "core_range_set": "{[0-0 - 1-1]}"}
  13 circular_buffer_allocate  {"address": 0,          "size": 131072,             "device_id": 1}
  14 circular_buffer_allocate  {"address": 0,          "size": 524288,             "device_id": 1}
  19 buffer_deallocate         {"address": 0,          "max_size_per_bank": 20480, "type": "L1"}

--- NORMAL ---
   8 buffer_allocate           {"address": 1552384,    "max_size_per_bank": 20480, "type": "L1",   "device_id": 1}
  12 circular_buffer_allocate  {"address": 111616,     "size": 131072,             "device_id": 0, "core_range_set": "{[0-0 - 1-1]}"}
  13 circular_buffer_allocate  {"address": 242688,     "size": 131072,             "device_id": 0}
  14 circular_buffer_allocate  {"address": 373760,     "size": 524288,             "device_id": 0}
  16 buffer_allocate           {"address": 4278185984, "max_size_per_bank": 4096,  "type": "DRAM", "size": 22528}
  21 buffer_deallocate         {"address": 1552384,    "max_size_per_bank": 20480, "type": "L1"}
```

Every `size` is identical — the point of `NO_DISPATCH`. What differs:

- Addresses become real, and the CBs are contiguous: 111616 -> 242688 -> 373760, exactly +131072.
- **An extra `buffer_allocate` at node 16 exists only in `NORMAL`** — a 22528-byte DRAM buffer for
  the program's kernel binaries / dispatch data, allocated at enqueue. DRAM, so it does not touch
  the L1 peaks, but it is why the traces are 24 vs 22 nodes.
- **`device_id` disagrees between node kinds in `NORMAL`**: CB nodes say `0`, buffer nodes say `1`.
  In `NO_DISPATCH` the CBs come from `track_program` with the `MeshDevice` (id 1); in `NORMAL` they
  come from `program.cpp` with the underlying `IDevice` (id 0). Do not key on `device_id` across
  node types.

---

## 6. `sharded_add.json` — tensors that are already sharded

The two inputs (nodes 3 and 5) carry **real, distinct addresses** (1540096, 1507328) even in
`NO_DISPATCH` — they were allocated before the window opened. Only the output has `address: 0`.

The output allocation shows the sharding:

```json
{ "counter": 9, "node_type": "buffer_allocate",
  "params": { "address": 0, "buffer_type": 1, "layout": "HEIGHT_SHARDED",
              "max_size_per_bank": 32768, "num_cores": 64, "page_size": 2048,
              "size": 2097152, "type": "L1" } }
```

2 MB total, 64 cores, 32768 B per core. Then all three CBs:

```json
{ "counter": 13, "node_type": "circular_buffer_allocate",
  "params": { "address": 0, "core_range_set": "{[0-0 - 10-9]}", "device_id": 1,
              "globally_allocated": "1", "size": 32768 } }
{ "counter": 14, "node_type": "circular_buffer_allocate",
  "params": { "...": "identical", "globally_allocated": "1", "size": 32768 } }
{ "counter": 15, "node_type": "circular_buffer_allocate",
  "params": { "...": "identical", "globally_allocated": "1", "size": 32768 } }
```

**Every CB is tensor-backed**, each exactly the 32768-byte shard: input a, input b, output.
`binary_ng` binds `c_0` / `c_1` / `c_2` straight to the operand buffers
(`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp:1437`), so the op
allocates no fresh CB memory. `peak_cb` is **0** and that is correct, not a bug.

Two traps:

- The 32768 on each CB equals the shard's `max_size_per_bank`, so the exclusion is exactly a
  de-duplication, not a loss.
- The CB's `core_range_set` is `{[0-0 - 10-9]}` = **110 cores**, while the shard's `num_cores` is
  **64**. The op declares the CB over the whole grid it may run on and repoints it at the shard.
  **You cannot read the shard grid off a CB node.**

Consequence for pricing: the honest cost here is 3 x 32 KB of operand shards, and the trace only
attributes the one that was allocated inside the window. `test_capture_window_bounds_peak_l1`
measures the same op at `1 x shard` with inputs created outside and `3 x shard` inside.

---

## 7. `sharded_add_in_place.json` — in-place

`ttnn.add(a, b, output_tensor=a)`. 16 nodes instead of 23. The aliasing is visible in two nodes:

```json
{ "counter": 2, "node_type": "function_start",
  "params": { "inputs": "2", "name": "BinaryNgDeviceOperation" },
  "input_tensors": [3, 5, 7],
  "connections": [4, 6, 8, 9, 10, 11, 12] }

{ "counter": 4, "node_type": "buffer",
  "params": { "buffer_type": 1, "layout": "HEIGHT_SHARDED", "size": 2097152, "type": "L1" },
  "connections": [3, 7, 13] }
```

`input_tensors` has three entries, not two: a, b, and a-as-output. And buffer node 4 has three
tensor nodes hanging off it. Nodes 3, 7 and 13 all carry `"address": 1474560, "tensor_id": 13` —
the input, the preallocated output argument and the returned tensor are the same buffer.
`add_buffer`'s `unique_id` de-dup is what collapses them onto one buffer node.

There is **no `buffer_allocate` node anywhere in the trace** — it goes from
`circular_buffer_deallocate_all` at 8 straight to the CBs at 9/10/11, with no
`create_device_tensor` frame at all. The three CBs are still `32768` each and still
`globally_allocated: "1"`, so the op moves 96 KB/core of live L1 while `peak_l1` reports **0**.
That gap is the capture window, not the accounting.

(That aliasing is also why that factory re-points tensor-backed CBs by `CBIndex` rather than by
enumeration order: the output CB must track the output buffer even when it shares a `Buffer*` with
an input.)

---

## 8. `conv2d.json` — a composite op

186 nodes. `python3 inspect_trace.py conv2d.json --view census`:

```
  43  tensor                             13  buffer_allocate
  38  function_start                     13  buffer_deallocate
  38  function_end                        4  circular_buffer_deallocate_all
  20  circular_buffer_allocate            1  dataflow_buffer_allocate
  14  buffer                              1  capture_start / 1 capture_end
```

`--view skeleton`, abridged:

```
   1 > ttnn.conv2d                                    (level 1)
   2   > Tensor::reshape                              (level 2)
   7   > tt::tt_metal::create_device_tensor
  12   > Tensor::deallocate
  15   > Tensor::reshape
  19   > tt::tt_metal::create_device_tensor
  24   > InterleavedToShardedDeviceOperation                   441522 ns
  32   > Tensor::to_layout                            \
  36   > tt::tt_metal::to_dtype                        |  weight + bias prep
  40   > Tensor::to_device                             |  (all DRAM)
  46   > Tensor::pad                                   |
  50   > Tensor::to_layout                             |
  54   > Tensor::to_device                            /
  60   > HaloDeviceOperation                                  3935600 ns
  62     > tt::tt_metal::create_device_tensor          (level 3)   <- halo output shard
  67     > Tensor::to_device   x4                                 <- halo config tensors
 100     > Tensor::deallocate  x4                                 <- config tensors freed
 112   < HaloDeviceOperation
 114   > Tensor::deallocate                                       <- sharded activation
 117   > Tensor::deallocate
 120   > tt::tt_metal::create_device_tensor
 127   > Conv2dDeviceOperation                                 709468 ns
 131     > tt::tt_metal::create_device_tensor          (level 3)   <- conv output shard
 136     > Tensor::to_device
 156   < Conv2dDeviceOperation
 161   > ShardedToInterleavedDeviceOperation                  3619516 ns
 163     > tt::tt_metal::create_device_tensor
 181 < ttnn.conv2d                                           11386987 ns
```

`--view l1` gives every L1-moving node with params verbatim. The highlights:

```
 21 [create_device_tensor]  buffer_allocate  {"buffer_type": 1, "layout": "HEIGHT_SHARDED", "max_size_per_bank": 10240, "num_cores": 103, "page_size": 64, "size": 1054720, "type": "L1"}
 27 [I2S]                   circular_buffer_deallocate_all
 28 [I2S]                   circular_buffer_allocate {"core_range_set": "{[0-0 - 10-8], [0-9 - 3-9]}", "globally_allocated": "1", "size": 10240}
 29 [I2S]                   circular_buffer_allocate {"core_range_set": "{[0-0 - 10-8], [0-9 - 3-9]}", "globally_allocated": "0", "size": 512}
 43 [Tensor::to_device]     buffer_allocate  {"buffer_type": 0, "max_size_per_bank": 6144, "size": 36864, "type": "DRAM"}   <- weights
 57 [Tensor::to_device]     buffer_allocate  {"buffer_type": 0, "max_size_per_bank": 2048, "size": 4096,  "type": "DRAM"}   <- bias
 64 [create_device_tensor]  buffer_allocate  {"buffer_type": 1, "max_size_per_bank": 27136, "num_cores": 103, "size": 2795008, "type": "L1"}  <- halo out
 70 [Tensor::to_device]     buffer_allocate  {"buffer_type": 3, "max_size_per_bank": 16, "page_size": 12, "size": 1236, "type": "L1"}
 76 [Tensor::to_device]     buffer_allocate  {"buffer_type": 3, "max_size_per_bank": 16, "size": 1236, "type": "L1"}
 82 [Tensor::to_device]     buffer_allocate  {"buffer_type": 3, "max_size_per_bank": 48, "size": 4120, "type": "L1"}
 88 [Tensor::to_device]     buffer_allocate  {"buffer_type": 3, "max_size_per_bank": 48, "size": 4120, "type": "L1"}
 91 [Halo]                  circular_buffer_deallocate_all
 92 [Halo]                  circular_buffer_allocate {"globally_allocated": "1", "size": 10240}
 93 [Halo]                  circular_buffer_allocate {"globally_allocated": "1", "size": 27136}
 94 [Halo]                  circular_buffer_allocate {"globally_allocated": "0", "size": 64}
 95 [Halo]                  circular_buffer_allocate {"globally_allocated": "0", "size": 64}
 96 [Halo]                  circular_buffer_allocate {"globally_allocated": "1", "size": 12}
 97 [Halo]                  circular_buffer_allocate {"globally_allocated": "1", "size": 12}
 98 [Halo]                  circular_buffer_allocate {"globally_allocated": "1", "size": 40}
 99 [Halo]                  circular_buffer_allocate {"globally_allocated": "1", "size": 40}
101 [Tensor::deallocate] x4 buffer_deallocate {"buffer_type": 3, ... 1236 / 1236 / 4120 / 4120 ...}
115 [Tensor::deallocate]    buffer_deallocate {"max_size_per_bank": 10240}   <- sharded activation
118 [Tensor::deallocate]    buffer_deallocate {"max_size_per_bank": 27136}   <- halo out, then...
122 [create_device_tensor]  buffer_allocate  {"max_size_per_bank": 27136}    <- ...reallocated
133 [create_device_tensor]  buffer_allocate  {"buffer_type": 1, "max_size_per_bank": 20480, "num_cores": 103, "page_size": 2048, "size": 2097152, "type": "L1"}  <- conv out
139 [Tensor::to_device]     buffer_allocate  {"buffer_type": 3, "max_size_per_bank": 32, "size": 2060, "type": "L1"}
142 [Conv2d]                circular_buffer_deallocate_all
143 [Conv2d]                circular_buffer_allocate {"core_range_set": "{[0-0 - 10-9]}", "globally_allocated": "0", "size": 12288}
144 [Conv2d]                circular_buffer_allocate {"globally_allocated": "1", "size": 20480}
145 [Conv2d]                circular_buffer_allocate {"globally_allocated": "0", "size": 18432}
146 [Conv2d]                circular_buffer_allocate {"globally_allocated": "0", "size": 12288}
147 [Conv2d]                circular_buffer_allocate {"globally_allocated": "0", "size": 30720}
148 [Conv2d]                circular_buffer_allocate {"globally_allocated": "0", "size": 4096}
149 [Conv2d]                circular_buffer_allocate {"globally_allocated": "1", "size": 20480}
150 [Conv2d]                circular_buffer_allocate {"globally_allocated": "1", "size": 20}
151 [Conv2d]                circular_buffer_allocate {"globally_allocated": "0", "size": 64}
152 [Conv2d]                circular_buffer_allocate {"globally_allocated": "1", "size": 27136}
165 [create_device_tensor]  buffer_allocate  {"buffer_type": 0, "max_size_per_bank": 262144, "type": "DRAM"}
168 [S2I]                   circular_buffer_deallocate_all
169 [S2I]                   dataflow_buffer_allocate {"address": 0, "borrows_memory": "1", "core_range_set": "{[0-0 - 10-8], [0-9 - 3-9]}", "size": 20480}
173 [Tensor::deallocate]    buffer_deallocate {"max_size_per_bank": 20480}
```

What this makes concrete:

- **Four `circular_buffer_deallocate_all` for four child programs** (I2S, Halo, Conv2d, S2I). Each
  opens its own program-scope window, which is what makes per-child CB totals separable rather
  than one blended figure.
- **Conv2d's 10 CBs split 6 real / 4 tensor-backed.** The real ones
  (12288 + 18432 + 12288 + 30720 + 4096 + 64 = 77888 B = 76.06 KB) are the entire `peak_cb`. The
  tensor-backed ones (20480 twice, 27136, 20) are the conv output, the reshuffled weights and the
  halo output — already in `peak_l1`.
- **Halo is 8 CBs of which only two are real, at 64 B each.** Its cost is almost entirely tensors:
  a 27136 B/core output shard plus four config tensors.
- **`buffer_type: 3` is `L1_SMALL`.** Halo's config tensors are L1_SMALL, but the node reports
  `"type": "L1"`, and the extractor only skips on `type == "DRAM"` — so **L1_SMALL bytes are folded
  into `peak_l1`** even though they live in a separate allocator region. Tiny here (16-48 B/core),
  but that is where a discrepancy against the allocator would come from.
- **`ShardedToInterleavedDeviceOperation` is already a Metal 2.0 op**: zero CBs, one
  `dataflow_buffer_allocate` with `borrows_memory: "1"`. It contributes 0 to `peak_cb` *and* 0 to
  `peak_dataflow_buffer`, because the buffer is a view onto the sharded tensor already counted. A
  ported op reporting `peak_cb == 0` is correct, not broken.
- **Nodes 118 and 122**: the halo output is deallocated at 27136 and immediately reallocated at
  27136 — `reallocate` defragmenting between halo and conv. Two events, net zero, invisible in the
  peak.
- **Core ranges differ between children**: the sharded ops use `{[0-0 - 10-8], [0-9 - 3-9]}`
  (103 cores, matching `num_cores: 103` on the shards) while Conv2d uses `{[0-0 - 10-9]}` (110).
  Within any one program every CB shares a range, which is why the flat `peak_cb` happens to be
  exact here.

Totals: `peak_cb` 76.06 KB, `peak_l1` 46.53 KB, `peak_total` 122.59 KB, and the peak lands inside
`Conv2dDeviceOperation` — Halo's own window never exceeds 37.12 KB. If this conv did not fit,
Conv2d's 76 KB of CBs is the thing to attack.

---

## 9. Gotcha checklist

1. The capture is a window. Inputs allocated before `begin_graph_capture` are invisible to
   `peak_l1`.
2. `globally_allocated` and `borrows_memory` are JSON **strings**, not ints.
3. `size` vs `max_size_per_bank` on buffer nodes — only the latter is per core.
4. `peak_cb` ignores `core_range_set`; it is a worst case, not a per-core truth.
5. A CB's `core_range_set` is the program's core range, not the tensor's shard grid.
6. `device_id` is not consistent across node types in `NORMAL` mode.
7. `NORMAL` mode adds a DRAM buffer for kernel binaries that `NO_DISPATCH` does not.
8. L1_SMALL buffers report `type: "L1"` and land in `peak_l1`.
9. `peak_cb == 0` is a valid answer for a fully tensor-backed sharded op and for a Metal 2.0 port.
10. Disable `enable_fast_runtime_mode` if you want the full op nesting; without it you still get
    the device-op nodes but not the composite ttnn frames.
11. **Freeing a buffer that was allocated before the capture window corrupts the peaks.** The
    running totals in `extract_resource_usage_per_core` are `size_t` with no floor on the
    decrement (`graph_trace_utils.cpp:366-368`), so a `buffer_deallocate` whose matching
    `buffer_allocate` is outside the window underflows. Measured on device:

    | scenario | true `peak_l1` | reported |
    |---|---|---|
    | free 32768/core (pre-window), then allocate 32768/core | 32768 | **0** (wrap cancels exactly) |
    | free 65536/core (pre-window), then allocate 32768/core | 32768 | **18446744073709518848** (= 2^64 - 32768) |

    Metal notices and logs `Can't hook deallocation of a buffer which allocation wasn't hooked`
    (`graph_tracking.cpp:178`), but the node is still emitted and still subtracted. Any op that
    frees an input (`deallocate_activation=True`, `ttnn.deallocate`, `reallocate`) hits this.
    **Mitigation: allocate every tensor the op may free inside the capture window**, and assert
    `peak_total < 2**32` as a tripwire.

---

## 10. Where `peak_l1` and `peak_cb` actually come from

Section 3 says the extractor sums them. This section is the provenance of each number, from the
program factory down to the JSON field.

### `peak_cb` <- `size` on `circular_buffer_allocate` <- `CircularBufferConfig::total_size()`

```
op's program factory
  uint32_t in0_CB_tiles = out_block_h * in0_block_w;              // block arithmetic
  if (B * num_blocks > 1) in0_CB_tiles *= MCAST_INPUT_BUFFERING_DEPTH;   // double buffering
  uint32_t in0_CB_size  = in0_CB_tiles * in0_aligned_tile_size;   // tiles -> bytes
  CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
      -> CircularBufferConfig::total_size_
                (ttnn/cpp/ttnn/operations/matmul/device/factory/
                 matmul_multicore_reuse_mcast_2d_program_factory.cpp:152-171, 2478)

CircularBufferImpl::size()   { return config_.total_size(); }     (tt_metal/impl/buffers/circular_buffer.hpp:39)
CircularBuffer::size()       { return impl_->size(); }            (circular_buffer.cpp:198)

GraphProcessor::track_program                                     (graph_processor.cpp:415)
  for (cb : program->circular_buffers())
      track_allocate_cb(cb->core_ranges(), 0, cb->size(), cb->globally_allocated(), device)

  -> node { "node_type": "circular_buffer_allocate", "params": { "size": <in0_CB_size>, ... } }

extract_resource_usage_per_core                                   (graph_trace_utils.cpp:325-333)
  if (!globally_allocated) { current_cb += size; peak_cb = max(peak_cb, current_cb); }
```

So `peak_cb` is **the op author's block arithmetic, verbatim**. It is per core by construction —
a `CircularBufferConfig` total size is what each core in the CB's `core_range_set` reserves. There
is no division, no allocator involvement, and nothing measured.

Check it against `matmul_l1_out.json`: 1024x1024 bf16 is 32x32 tiles, the chosen grid is 2x2
(`core_range_set: "{[0-0 - 1-1]}"`), so `per_core_M = per_core_N = 32/2 = 16` and the output block
is `16 * 16 = 256` tiles. A bf16 tile is 2048 B, so `256 * 2048 = 524288` — exactly the third CB
in the trace. The other two are `64 * 2048 = 131072` each.

In `NORMAL` mode the same node is emitted from a different place — `program.cpp:1693`, during real
CB allocation, which is why it carries a real `address` and the underlying `IDevice` id. The `size`
is identical because it is the same `config_.total_size()`.

### `peak_l1` <- `max_size_per_bank` on `buffer_allocate` <- `Buffer::aligned_size_per_bank()`

```
Buffer::aligned_page_size()      = align(page_size(), alignment())            (buffer.cpp:764)
Buffer::aligned_size()           = num_dev_pages() * aligned_page_size()      (buffer.cpp:765)
Buffer::aligned_size_per_bank()  = calculate_bank_size_spread(                (buffer.cpp:767)
                                       aligned_size(), aligned_page_size(),
                                       num_banks, alignment())
    where num_banks = sharded ? num_cores() : allocator->get_num_banks(type)

detail::calculate_bank_size_spread                                (allocator.cpp:609)
    num_pages                     = size_bytes / page_size_bytes
    num_equally_distributed_pages = 1 + ((num_pages - 1) / num_banks)   // ceil-div
    return num_equally_distributed_pages * round_up(page_size_bytes, alignment_bytes)

GraphProcessor::track_allocate                                    (graph_processor.cpp:224-236)
  uint32_t max_size_per_bank = buffer->aligned_size_per_bank();
  -> node { "node_type": "buffer_allocate", "params": { "max_size_per_bank": ..., ... } }

extract_resource_usage_per_core                                   (graph_trace_utils.cpp:353-370)
  if (type != "DRAM") { current_l1 += max_size_per_bank; peak_l1 = max(peak_l1, current_l1); }
```

Two things follow from that ceil-div:

- **It is a worst case per bank, not an average.** 100 pages over 64 banks gives
  `1 + 99/64 = 2` pages per bank, i.e. 2 pages charged to every bank even though 28 banks hold
  only one. That is correct for an "does it fit on the worst core" question.
- **For a sharded buffer `num_banks` is `num_cores()`, not the allocator's bank count**, so a
  sharded `max_size_per_bank` is exactly one shard (rounded to page alignment). In
  `sharded_add.json` that is `32768`, and the tensor-backed CBs report the same `32768` — which is
  why excluding them is a de-duplication and not a loss.

`page_size` and `layout` in the same node let you sanity-check the figure, and `num_cores` is `0`
for interleaved (`buffer->num_cores().value_or(0)`).

Note this is the *allocator's* view of the buffer, so it includes page-alignment padding that the
logical tensor does not have. That is why `extract_l1_output_buffer_allocation_size_per_core`
(`graph_trace_utils.cpp:278`) also uses `aligned_size_per_bank()` rather than `buffer->size()`,
with a comment saying `size()` underestimates sharded buffers whose shard rows are tile-padded.

### The other `peak` function, and why it is different

`ttnn.graph.extract_peak_L1_memory_usage` (`graph_trace_utils.cpp:60`) is **not** the per-core
figure and is not a component of `peak_total`. It keeps its own running totals and, crucially,
adds buffers by their **whole-buffer** `size` rather than `max_size_per_bank`:

```cpp
} else if (v[kNodeType] == kNodeBufferAllocate && v[kParams][kType] == "L1") {
    total_buffer += json_to_int(v[kParams][kSize]);      // whole buffer, ALL banks
...
peak_memory_usage = std::max(peak_memory_usage, total_cb + total_buffer);
```

so it mixes device-wide tensor bytes with per-core CB bytes in one sum. Treat it as a rough
device-level aggregate; for anything "does this fit in L1" use `extract_resource_usage_per_core`.
It also folds scratchpads into `total_cb` and walks `connections` to find the buffer on a
deallocate, rather than reading the dealloc node's own params.

### Everything that reports a peak

| API | scope | source of truth |
|---|---|---|
| `extract_resource_usage_per_core(trace)` | per core, 5 fields | `size` (CB/DFB/scratchpad) + `max_size_per_bank` (L1 buffers) |
| `extract_peak_L1_memory_usage(trace)` | device-wide-ish aggregate, mixed units | `size` on both kinds |
| `extract_circular_buffers_peak_size_per_core(trace)` | deprecated | delegates, returns `.peak_cb` (`:295`) |
| `extract_l1_buffer_allocation_peak_size_per_core(trace)` | deprecated | delegates, returns `.peak_l1` (`:291`) |
| `extract_peak_memory_usage(trace)` | deprecated | delegates, returns `.peak_total` (`:304`) |
| `extract_l1_output_buffer_allocation_size_per_core(tensor)` | one tensor, per core | `Buffer::aligned_size_per_bank()` directly, no trace |
| `tt::tt_metal::calculate_total_cb_size(program)` | per core, CBs only | sums `cb->size()` skipping `globally_allocated()`, i.e. `peak_cb` for one program (`ttnn/cpp/ttnn/operations/cb_utils.hpp:54`) |
| `Device::get_total_cb_allocated()` | live device-wide | merges `get_cb_l1_regions_per_core()` across registered programs, de-duplicating reused addresses (`tt_metal/impl/device/device.cpp:1107`) |
