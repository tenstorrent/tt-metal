# `all_gather_minimal_matmul_async` — Internals Reference

This document covers the fused all-gather + matmul op used in Wan2.2 tensor-parallel linear layers. It is meant as a guide for someone modifying the op or debugging it.

---

## 1. What the Op Does

`all_gather_minimal_matmul_async` fuses two operations that would otherwise be sequential:

1. **All-gather along K** — each device holds a shard of the activation tensor (shape `[M, K/ring_size]`). The all-gather reconstructs the full `[M, K]` across all devices.
2. **Matmul** — `[M, K] × [K, N]` → `[M, N]`, with optional bias add, activation, and addcmul post-processing.

The key insight is that the matmul can begin consuming the first K-block of gathered data while subsequent K-blocks are still being received over the ring — **compute and communication overlap**.

Optionally fuses:
- Bias add (`+ bias[N]`)
- Unary activation (e.g. GELU, SiLU)
- Addcmul: `out = ternary_a + scalar × matmul_out × ternary_b`

Returns a list of output tensors (length 1 normally, or `chunks` if output splitting is used).

---

## 2. Source File Map

| Layer | File |
|---|---|
| Public Python API | `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/all_gather_minimal_matmul_async.hpp` |
| Public C++ wrapper | `...all_gather_minimal_matmul_async.cpp` |
| Python nanobind | `...all_gather_minimal_matmul_async_nanobind.cpp` |
| Device op struct | `.../device/all_gather_minimal_matmul_async_device_operation.hpp/.cpp` |
| Types (params/inputs) | `.../device/all_gather_minimal_matmul_async_device_operation_types.hpp` |
| Program factory | `.../device/all_gather_minimal_matmul_async_program_factory.hpp/.cpp` |
| Kernel: in0 DM | `.../device/kernels/dm_in0_sender.cpp` |
| Kernel: in1 DM + output | `.../device/kernels/dm_in1_sender_out.cpp` |
| Kernel: compute | `.../device/kernels/compute.cpp` |
| Shared DM utilities | `.../device/kernels/matmul_dataflow_common.hpp` |
| Test | `models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py` |

All paths are relative to the tt-metal repo root. The main directory is:

```
ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/
```

---

## 3. Fusion Pattern: How All-Gather and Matmul Overlap

### K-dimension chunking

The activation K dimension is divided into `K_blocks = padded_K_tiles / K_block_tiles` blocks. Each device owns `K_blocks_per_device = K_blocks / ring_size` blocks locally. The remaining `(ring_size - 1) × K_blocks_per_device` blocks arrive over the ring in `ring_size - 1` steps.

```
Device 0 local K:  [K₀₀, K₀₁, ...]
                     ↓ (already in L1)
Compute iteration 0: matmul K₀₀..K₀ₙ₋₁ ← no waiting needed

Device 1 → Device 0: [K₁₀, K₁₁, ...]
                     ↓ (arrives over ring)
Compute iteration 1: matmul K₁₀..K₁ₙ₋₁ ← overlapped with Device 2→0 transfer
...
```

The in0 sender kernel coordinates this: it reads local K-blocks first, then receives remote blocks from the ring and feeds them into the circular buffer consumed by compute. Compute never has to wait for the whole gather to complete before starting work.

### Ring direction

The ring uses a **linear topology** (not wrap-around ring) with both forward and backward directions. `num_targets_forward_direction` and `num_targets_backward_direction` are set based on `ring_index` so each device knows how many other devices it needs to reach/receive from.

### Persistent ping-pong buffers

The all-gather uses pre-allocated persistent buffers (passed as `persistent_output_buffer`). These are separate from the actual output and act as staging area for gathered activations. This avoids repeated allocation overhead across iterations.

---

## 4. Core Grid Layout

### Parallelization axes

The core grid is a 2D rectangle `(grid_x × grid_y)`. The matmul parallelizes:
- **M dimension** along one axis (M cores compute different output rows)
- **N dimension** along the other axis (N cores compute different output columns)

Which axis is M vs N is controlled by `force_transpose` and the auto-detected `transpose_core_grid` flag (set when M > N after blocking):

| `transpose_core_grid` | M axis | N axis | in0 DM on | in1 DM on |
|---|---|---|---|---|
| `false` (default) | `grid_y` (rows) | `grid_x` (columns) | NOC1 / RISCV1 | NOC0 / RISCV0 |
| `true` | `grid_x` (columns) | `grid_y` (rows) | NOC0 / RISCV0 | NOC1 / RISCV1 |

### Core ranges within the grid

```
┌─────────────────────────────────────┐
│  in0_sender_cores (first row/col)   │  ← reads local in0, mcasts + ring sends
├─────────────────────────────────────┤
│  in0_receiver_no_fabric_cores       │  ← receives in0 via NOC mcast (local)
├─────────────────────────────────────┤
│  in0_receiver_fabric_cores          │  ← receives in0 from remote ring devices
│  (last 2–3 rows/cols)               │    via fabric mux
├─────────────────────────────────────┤
│  mux_cores (bottom edge)            │  ← fabric mux: handles inter-device comms
└─────────────────────────────────────┘
```

All cores in the grid also run the **compute kernel**. The in1 (weight) DM kernel multicast broadcast is separate: a single `in1_sender_core` reads weights and mcasts to `in1_receiver_cores` (the rest of the grid along the N axis).

---

## 5. Kernel Roles

Five kernel instances are deployed (from two source files):

### `dm_in0_sender.cpp` — 3 deployments

| Instance | Core range | Role |
|---|---|---|
| `in0_sender_kernels` | First row/column | Reads local activation shard, sends to receivers within device, and to ring neighbors |
| `in0_receiver_no_fabric_kernels` | Middle rows/columns | Receives in0 via on-chip NOC mcast, feeds `in0_cb` |
| `in0_receiver_fabric_kernels` | Last 2–3 rows/cols | Receives in0 from remote devices via fabric mux |

The distinction between fabric/no-fabric receivers is whether a core is connected to the inter-device fabric (Ethernet/mux). Fabric cores handle the ring-level receive path.

### `dm_in1_sender_out.cpp` — 2 deployments

| Instance | Core range | Role |
|---|---|---|
| `in1_sender_kernels` | First core on N axis | Reads weight tiles from DRAM, mcasts to all N cores, writes output to DRAM |
| `in1_receiver_kernels` | Remaining N cores | Waits for in1 mcast, pushes to `in1_cb`, writes its output slice to DRAM |

One of the in0 or in1 kernels (depending on `transpose_core_grid`) also acts as the **output writer**: it pops from the output CB and writes to DRAM after each K-pass completes.

### `compute.cpp` — 1 deployment on all cores

Runs the actual matmul. Waits for `in0_cb` and `in1_cb` to be populated, iterates over M-blocks × N-blocks × K-blocks, calls `matmul_tiles` with the configured subblock dimensions, and pushes results into `out_cb`. Also handles fused bias, addcmul, and activation post-processing.

---

## 6. Data Movement: Circular Buffers

All cores allocate the following CBs:

| CB | Purpose | Double-buffered? |
|---|---|---|
| `c_0` (in0_cb) | Activation tiles | Yes (2×) — hides transfer latency |
| `c_1` (in1_cb) | Weight tiles | Yes (2×) |
| `c_2` (out_cb) | Output tiles | Yes (2×) |
| `c_3` (intermediate_cb) | Matmul accumulator (DST) | No |
| `c_4` (in2_cb) | Bias tiles (if `FUSE_BIAS`) | No |
| `c_5` (ternary_a_cb) | Addcmul input A | No |
| `c_6` (ternary_b_cb) | Addcmul input B | No |

### Output write deferral

To reduce NOC congestion, the output DM kernel uses a `defer_write_k_block` mechanism: it delays flushing output tiles to DRAM by one K-block step, so that write bursts are staggered across cores rather than all hitting at the same time.

---

## 7. Synchronization

### Intra-device (NOC semaphores)

The in0 sender/receiver handshake uses three semaphores per producer-consumer pair:

```
sender:   sem_send  → atomically inc on receiver; dec after mcast
receiver: sem_recv  → waits until incremented (data ready)
          sem_valid → signals back to sender that receiver consumed data
```

This prevents the sender from overwriting a buffer before the receiver has read it.

### Inter-device (fabric mux)

The fabric mux kernels maintain a connection state machine per link. Each device has:
- `forward_coord` / `backward_coord`: neighbor device coordinates
- Per-link unicast routing tables passed at runtime
- `num_buffers_per_channel` ring buffers for in-flight data

The `multi_device_global_semaphore` (a vector of 2 `GlobalSemaphore` objects) coordinates the start of each all-gather step across the ring.

### Barrier semaphore

`barrier_semaphore` is an optional additional global semaphore used for a full barrier across devices before the op begins. This is needed when the previous op on a different device has not yet completed.

### CB-level synchronization

Within a core, the standard CB `reserve_back` / `push_back` (producer) and `wait_front` / `pop_front` (consumer) protocol ensures correct ordering between DM kernels and compute.

---

## 8. User-Controlled Parameters

### `MinimalMatmulConfig` — tiling and grid

```python
ttnn.MinimalMatmulConfig(
    M_block_size=8,          # M tiles processed per core per K-pass (output height block)
    K_block_size=8,          # K tiles consumed per matmul block iteration
    N_block_size=8,          # N tiles processed per core (output width block)
    subblock_h=2,            # Sub-block height within M_block (must divide M_block_size)
    subblock_w=2,            # Sub-block width within N_block (must divide N_block_size)
    compute_with_storage_grid_size=core_grid,  # ttnn.CoreGrid(x=..., y=...)
)
```

Larger blocks improve reuse (fewer DRAM reads) but require more L1. Subblock dimensions must satisfy `subblock_h × subblock_w ≤ 8` (DST register limit).

### CCL / ring parameters

| Parameter | Type | Effect |
|---|---|---|
| `multi_device_global_semaphore` | `list[GlobalSemaphore]` (length 2) | Semaphores for CCL step synchronization — must be pre-allocated with `create_global_semaphores` |
| `num_links` | `int` | Number of parallel Ethernet/fabric links per device for the all-gather. Higher = more bandwidth. |
| `num_workers_per_link` | `int` | Worker cores assigned per link. More workers → more parallelism per link. |
| `num_buffers_per_channel` | `int` | Ring buffers per channel (flow control / in-flight depth). 48 is typical. |
| `topology` | `ttnn.Topology` | `Ring` (wrap-around) or `Linear` (end-to-end). Always `Linear` in Wan2.2. |
| `cluster_axis` | `int` (0 or 1) | Which axis of the device mesh to all-gather along. 0 = row axis, 1 = column axis. |
| `barrier_semaphore` | `GlobalSemaphore` or `None` | Optional cross-device barrier before op start. |

### Matmul compute config

```python
ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,  # HiFi4 for more precision, LoFi for speed
    math_approx_mode=False,
    fp32_dest_acc_en=True,   # Accumulate in fp32 (important for bf16 inputs)
    packer_l1_acc=True,      # L1 accumulation across K-blocks
)
```

`packer_l1_acc=True` enables accumulation in L1 across K-blocks, which is critical for correctness when K is split. **Do not disable** unless you know the K dimension is small enough to fit in one block.

### Optional fused post-processing

| Parameter | Type | Effect |
|---|---|---|
| `bias_tensor` | `Tensor [1, N]` or `None` | Add bias after matmul: `out += bias` |
| `fused_activation` | `UnaryWithParam` or `None` | Apply unary op (GELU, SiLU, etc.) after bias |
| `scalar` | `float` or `None` | Scalar for addcmul: enables `a + scalar × matmul × b` |
| `addcmul_input_tensor1` | `Tensor [M, N]` or `None` | `a` in addcmul (residual/base) |
| `addcmul_input_tensor2` | `Tensor [1, N]` or `[M, N]` or `None` | `b` in addcmul (gate). If shape is `[1, N]`, it is broadcast across M. |

### Output splitting

| Parameter | Default | Effect |
|---|---|---|
| `chunks` | `1` | Split output tensor into `N` equal chunks along `dim` |
| `dim` | `-1` | Dimension to split (default = last dim = N) |

When `chunks > 1`, the op returns a list of `chunks` tensors instead of one.

### Other flags

| Parameter | Default | Effect |
|---|---|---|
| `persistent_output_buffer` | `None` | Pre-allocated staging buffer for gathered activations. Avoids allocation on every call. |
| `force_transpose` | `False` | Override the auto-detected transpose_core_grid flag. Useful for non-square grids where the auto heuristic is wrong. |

---

## 9. Fused Operation Details

### Bias add (compile-time flag `FUSE_BIAS`)

Enabled when `bias_tensor` is provided. The bias is read once by the in1 sender, placed in `in2_cb`, and added in the compute kernel after the matmul accumulation. Bias is `[1, N]` (broadcast across all M rows).

### Unary activation (`fused_activation`)

Applied in the compute kernel immediately after bias add. The `UnaryWithParam` struct carries the op type and an optional float parameter. Common choices: `UnaryOpType::GELU`, `UnaryOpType::SILU`, `UnaryOpType::RELU`.

### Addcmul (`fused_ternary_scalar` + inputs)

Computes: `out = ternary_a + scalar × matmul_out × ternary_b`

- `ternary_a` is a residual tensor, same shape as output `[M, N]`
- `ternary_b` is a gate tensor, either `[M, N]` or `[1, N]` (broadcast)
- Scalar is passed as a runtime arg to the compute kernel (cast to `uint32` via `std::bit_cast`)

This pattern is used for gated linear unit (GLU) variants in diffusion transformer blocks.

---

## 10. How to Make Modifications

### Changing block sizes

Edit the `MinimalMatmulConfig` passed from Python. Block sizes affect both L1 usage and performance:
- Too large → L1 overflow → runtime error
- Too small → too many DRAM fetches → bandwidth bound
- Rule: `M_block_size × K_block_size × tile_size + K_block_size × N_block_size × tile_size` must fit in L1 per core

Subblock sizes must divide block sizes and satisfy the DST constraint (`subblock_h × subblock_w ≤ 8` for Wormhole).

### Adding a new fused post-op

1. Add a new tensor input to `AllGatherMinimalMatmulAsyncInputs` in `..._device_operation_types.hpp`
2. Add a corresponding `bool fuse_X` flag or `optional<...>` parameter to `AllGatherMinimalMatmulAsyncParams`
3. In the program factory (`..._program_factory.cpp`):
   - Allocate a new CB for the new tensor
   - Add a compile-time define `FUSE_X` if the tensor is present
   - Set runtime args on the appropriate DM kernel (in1 sender reads extra tensors)
4. In `dm_in1_sender_out.cpp`:
   - Read the new tensor and push to its CB
5. In `compute.cpp`:
   - After `#ifdef FUSE_X`, pop from the new CB and apply the operation

### Changing the ring topology

The topology is passed as `ttnn::ccl::Topology`. `Linear` avoids wrap-around and is required for mesh-attached devices. Switching to `Ring` requires changing the forward/backward target count calculation in `..._program_factory.cpp` (search for `num_targets_forward_direction`).

### Debugging semaphore hangs

If the op hangs (device stuck), it is almost always a semaphore deadlock. Steps:
1. Check that `multi_device_global_semaphore` has exactly 2 entries and was created with `create_global_semaphores(mesh_device, 2, ...)`
2. Verify `ring_size` matches the number of devices on the cluster axis
3. Check `num_links` does not exceed the number of physical Ethernet links available
4. If using `barrier_semaphore`, ensure it is created on the same device mesh as the main semaphores
5. Reduce `num_buffers_per_channel` (e.g. from 48 to 8) to reduce memory pressure as a diagnostic step

### Debugging numerical errors

1. Start by running the non-fused fallback (`all_gather_async` + `minimal_matmul` separately) and compare outputs with `assert_quality()` from the test file
2. Disable fused post-ops one by one (`bias_tensor=None`, `fused_activation=None`, `scalar=None`) to isolate which stage introduces error
3. Try `MathFidelity.HiFi4` to rule out precision issues
4. Check `fp32_dest_acc_en=True` — this is required when accumulating bf16 tiles across many K-blocks

---

## 11. Data Flow Summary Diagram

```
Per device (ring_size = R, device rank = i):

DRAM
 │
 ├─ input_tensor [M, K/R]   ─→ in0_sender_kernel
 │                                    │
 │                                    ├─ NOC mcast → in0_receiver (local cores)
 │                                    │
 │                                    └─ Fabric/Ethernet → next/prev ring devices
 │                                                               │
 │                                              in0_receiver_fabric (last rows/cols)
 │
 ├─ weight_tensor [K, N]    ─→ in1_sender_kernel
 │                                    │
 │                                    └─ NOC mcast → in1_receiver (all N cores)
 │
 └─ bias/addcmul tensors    ─→ in1_sender_kernel ─→ in2_cb / ternary CBs


All cores (M×N grid):
  in0_cb ──┐
           ├─→ compute_kernel ─→ out_cb ─→ (output writer DM) ─→ DRAM output
  in1_cb ──┘
           (optional: in2_cb for bias, ternary CBs for addcmul)
```

Each ring step: in0_sender streams one device's K-blocks through in0_cb into compute while the next device's K-blocks arrive over the ring. Compute accumulates across all K-blocks before writing the final output.
