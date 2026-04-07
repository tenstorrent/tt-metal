# NP + Conv3D Fusion Plan: `neighbor_pad_conv3d`

## Goal
Fuse NeighborPad (fabric_only) + Conv3D into one program object / one dispatch, eliminating
the two-dispatch overhead on BH Loud Box (8xP150). Modeled after `all_gather_matmul_async`.

---

## 1. Current State vs Target

| Layer | Before | After |
|-------|--------|-------|
| Python | Two calls: `neighbor_pad_halo_only(...)` then `conv3d(...)` | One call: `neighbor_pad_conv3d(...)` |
| Dispatch | Two `device_operation::launch<>` → two go_signals | One `device_operation::launch<>` |
| Sub-device | activate/deactivate halo sub-devices around NP | None needed (one program owns all cores) |
| Semaphore wiring | Python passes `input_progress_sem_addr` between ops | Factory allocates and wires at build time |
| Kernels | Unchanged | Unchanged |

---

## 2. Op Signature

New primitive: `ttnn::experimental::prim::NeighborPadConv3dDeviceOperation`

```cpp
struct operation_attributes_t {
    // NP side (subset of NeighborPadAsyncParams needed for fabric_only H-dim mode)
    NeighborPadAsyncParams np_params;        // contains padding, ring_size, cluster_axis,
                                            //   h_neighbor_semaphore, barrier_semaphore,
                                            //   progress_semaphore, progress_t_batch_size,
                                            //   fabric_only=true
    // Conv3d side
    Conv3dParams conv3d_params;             // contains config (with use_h_halo_buffer=true,
                                            //   input_progress_t_batch_size, sub_device_id=nullopt
                                            //   since we manage cores ourselves)
    // Fusion
    CoreCoord np_core_grid_offset;          // logical {0,0}: NP fabric cores start here
                                            // same role as all_gather_core_grid_offset L113
};

struct tensor_args_t {
    Tensor input;          // [B,T,H,W,C] on DRAM
    Tensor weight;         // conv3d weights
    std::optional<Tensor> bias;
    Tensor halo_buffer;    // compact halo buffer in DRAM (pre-allocated, ping-pong safe)
};
```

The public ttnn op (`ttnn::experimental::neighbor_pad_conv3d`) combines what today are
`NeighborPadAsyncParams` + `Conv3dConfig` into one op. The Python caller constructs one
`Conv3dConfig` with `use_h_halo_buffer=true` and all halo dims filled; the factory wires them.

---

## 3. Program Factory Structure

`neighbor_pad_conv3d_program_factory.cpp` — one `create()` that returns one `Program`:

```
1. Program program{};

2. Call NP factory helper (fabric_only path only):
       auto np_shared = build_np_fabric_only_program_artifacts(
           program, input, halo_buffer, np_params, np_core_grid_offset);
   — This is the analogue of build_all_gather_async_minimal_default_program_artifacts()
     at all_gather_matmul_async_program_factory.cpp:128.
   — Adds h_reader + h_writer kernels to program on cores {0,0}..{N_fabric-1, 0}.
   — Computes reader_noc_coords for the progress semaphore CRTA (lines 488-513 in NP factory).

3. Call Conv3d factory helper:
       auto conv3d_shared = build_conv3d_program_artifacts(
           program, input, weight, bias, output, conv3d_params, conv3d_core_range_set);
   — Adds reader_vol2col + compute + writer kernels to the SAME program.
   — core_range_set excludes NP fabric cores: full_grid.subtract(np_fabric_core_range).
   — Passes progress_sem_addr and halo_buffer_addr directly as runtime args (no Python round-trip).

4. return {program, {np_shared, conv3d_shared}};
```

Key pattern from all_gather_matmul: matmul helper is called first (lines 79-117), then
all_gather helper with the SAME program object (line 128). For us order is reversed:
NP first (because it needs to know conv3d reader core coords for the CRTA), then conv3d.

In practice this means NP helper must accept the conv3d CoreRangeSet so it can enumerate
reader NOC coords — same logic as NP factory lines 490-496 where `full_grid.subtract(worker_core_ranges)`
gives conv3d cores.

---

## 4. Core Grid Layout

NP fabric_only mode uses 2 cores per link, typically 2 links → 4 cores at logical (0..3, 0).
Conv3d gets the rest (e.g. 116 cores on BH P150 minus 4 = 112 compute cores).

```
np_fabric_core_range  = CoreRangeSet(CoreRange({0,0}, {2*num_np_links - 1, 0}))
conv3d_core_range_set = full_grid_coreset.subtract(np_fabric_core_range)
```

No sub-device manager. No sub_device_id anywhere. One program → one dispatch → both kernel
groups start at the same go_signal. This mirrors `all_gather_core_grid_offset` at
device_operation.cpp:212 / program_factory.cpp:43.

---

## 5. Semaphore Wiring

Today: Python allocates a `GlobalSemaphore`, passes `.address()` to NP as `progress_semaphore`
and to conv3d config as `input_progress_sem_addr`. Two separate dispatches.

After fusion, factory does it all at build time:

```
Step A (in create()):
    auto progress_sem = CreateSemaphore(program, conv3d_core_range_set, 0);
    // progress_sem is a per-core L1 semaphore ID, identical to the conv3d reduction semaphore
    // pattern at conv3d_program_factory.cpp:351.
    uint32_t progress_sem_l1_addr = GetSemaphoreAddr(program, progress_sem);

Step B (NP h_writer CRTA, built inside build_np_fabric_only_program_artifacts()):
    CRTA[4] = progress_sem_l1_addr   // same slot as NP factory line 503
    CRTA[5] = num_conv3d_reader_cores
    CRTA[6+] = {x,y} pairs for each conv3d reader core
    // NP kernel reads these at runtime: minimal_default_writer.cpp lines 427-430

Step C (conv3d reader runtime args, set inside build_conv3d_program_artifacts()):
    reader_args[11] = progress_sem_l1_addr   // input_progress_sem_addr
    reader_args[12] = input_progress_t_batch_size
    // reader_vol2col.cpp lines 545-546 reads these under CONV3D_INPUT_PROGRESS_SEM
```

No GlobalSemaphore needed — L1 semaphore created inside the single program suffices because
both kernel groups live in the same program and same dispatch.

Halo buffer address wired similarly:
```
    reader_args[13] = halo_buffer.buffer()->address()  // h_halo_buffer_addr
    // reader_vol2col.cpp lines 552-557
```

---

## 6. Runtime Args Override

`override_runtime_arguments()` combines both:

```cpp
void override_runtime_arguments(...) {
    // NP side: update input/output/semaphore addresses
    //   mirrors NP factory override_runtime_arguments() lines 84-124
    auto& hw = GetCommonRuntimeArgs(program, np_shared.h_writer_kernel_id);
    hw[0] = input.buffer()->address();
    hw[1] = halo_buffer.buffer()->address();  // output of NP = compact halo buffer
    // hw[4] = progress_sem_l1_addr — static after create(), no change needed

    // Conv3d side: update input/output/weight/bias + halo buffer addr
    //   mirrors Conv3dProgramFactory::override_runtime_arguments() lines 878-909
    for (uint32_t i = 0; i < conv3d_shared.num_cores; i++) {
        reader_args[0]  = input.buffer()->address();
        reader_args[11] = progress_sem_l1_addr;   // static but keep for ping-pong safety
        reader_args[13] = halo_buffer.buffer()->address();
        reader_args[14..18] = h_halo_outer_dim, H, W, padding_h, padding_w;
        writer_args[0] = output.buffer()->address();
        writer_args[1] = weight.buffer()->address();
        writer_args[2] = bias addr;
    }
}
```

---

## 7. Python API

Before:
```python
halo_tensor = ccl_manager.neighbor_pad_halo_only(x_BTHWC,
    padding_h=1, padding_w=1, progress_semaphore=prog_sem, ...)
x_out = ttnn.experimental.conv3d(x_BTHWC, weight, bias,
    config=conv_config_with_halo_fields, ...)
```

After:
```python
x_out = ttnn.experimental.neighbor_pad_conv3d(
    x_BTHWC, weight, bias,
    halo_buffer=halo_buf,          # pre-allocated compact DRAM buffer (same ping-pong scheme)
    np_padding_h=1, np_padding_w=1,
    np_semaphores=...,             # h_neighbor_semaphore, barrier_semaphore (fabric sync, not progress)
    conv_config=conv_config,       # Conv3dConfig without halo/progress fields (factory owns those)
    np_num_links=2,
)
```

The halo_buf ping-pong can remain in Python since it's a tensor allocation concern, not dispatch.

---

## 8. What Does NOT Change

- `reader_vol2col.cpp` — CONV3D_H_HALO and CONV3D_INPUT_PROGRESS_SEM code paths unchanged
- `minimal_default_writer.cpp` — NP_PROGRESS_SEM signaling unchanged (lines 412-434)
- Compact halo buffer layout: [H-top | H-bot | W-left | W-right] unchanged
- T-batch progress semaphore mechanism: NP writer signals conv3d readers via NOC atomic after
  each outer_dim batch, conv3d reader polls local L1 (noc_semaphore_wait_min) — unchanged
- DRAM as the handoff point between NP and conv3d — unchanged

---

## 9. Estimated Perf Gain

| Source | Estimate |
|--------|----------|
| Two go_signal sends eliminated | ~2-4ms |
| Python dispatch overhead (two `device_operation::launch`) | ~1-2ms |
| Sub-device activate/deactivate (activate_halo_sub_devices + deactivate) | ~2-4ms |
| **Total** | **~5-10ms per layer** |

On a 1.36s decode with ~20 conv3d layers in the VAE decoder, this is ~100-200ms potential
savings if all layers are fused (7-15% decode time). Even at 3-5 fused layers this is
meaningful.

---

## 10. Implementation Order

1. **Extract NP helper function** (`build_np_fabric_only_program_artifacts`) from
   `neighbor_pad_async_program_factory.cpp` that accepts an existing `Program&` and
   returns `NpSharedVariables`. Analogous to `build_all_gather_async_minimal_default_program_artifacts`.

2. **Extract conv3d helper function** (`build_conv3d_program_artifacts`) from
   `conv3d_program_factory.cpp` that accepts `Program&`, `CoreRangeSet`, and progress sem addr,
   returns `Conv3dSharedVariables`. The existing `create()` becomes a thin wrapper.

3. **Create new files**:
   - `ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/neighbor_pad_conv3d_device_operation.hpp/.cpp`
   - `ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/neighbor_pad_conv3d_program_factory.hpp/.cpp`
   - `ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/neighbor_pad_conv3d.cpp`
   - `ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/neighbor_pad_conv3d_nanobind.cpp`

4. **Wire program factory**: `create()` calls NP helper first (to get reader NOC coords),
   then conv3d helper with those coords folded into the progress semaphore CRTA.

5. **Add to CMakeLists** and register nanobind op under `ttnn.experimental`.

6. **Update `vae_wan2_1.py`**: replace the two-op sequence with one `neighbor_pad_conv3d` call.
   Remove the sub-device setup / activate/deactivate calls that were needed for concurrent dispatch.

7. **Validate**: run `test_conv3d_fused.py` (or equivalent) to confirm PCC + perf improvement.

---

## 11. Root Cause of Deadlock (SOLVED 2026-04-05)

### Bug
`neighbor_pad_conv3d_program_factory.cpp` hardcoded `np_dim = 1` with wrong comment
"H-dim is index 1 in BTHWC layout". H is actually at index **2** in BTHWC [B, T, H, W, C].

### Impact
With `np_dim=1`:
- `outer_dim_size = B = 1` (only batch dimension counted, not T)
- NP writer processed 1 outer_dim item and signaled progress semaphore ONCE
- Conv3d reader waited for signal #2 for the second T-block → DEADLOCK

With `np_dim=2` (correct):
- `outer_dim_size = B * T = N * T_latent` (e.g., 21 for 81 video frames)
- NP writer processes all T-slices, signaling after each `T_out_block` group
- Conv3d reader proceeds block by block as NP completes each T-batch → no deadlock

### Fix
`neighbor_pad_conv3d_program_factory.cpp` line 98: `constexpr uint32_t np_dim = 2;`

### Verification
- `run_vae_decoder_ablation.py` (2x4 BH LoudBox, 480p, 81 video frames):
  - Timed run: **1.358s** total VAE (upload 28ms + decode 1018ms + readback 312ms)
  - No deadlock
- `test_wan_decoder[2x4_h0_w1-bf16-no_cache_full_T-check_output-fake_weights-10f-480p]`:
  - **PASSED**: PCC = 99.9763% (threshold: 99.9%)

---

## 12. Debugging Lessons

1. **DEVICE_PRINT needs `TT_METAL_DEVICE_PRINT=1`** (not just `TT_METAL_DPRINT_CORES`). Without it, DEVICE_PRINT is a no-op and "no output" doesn't mean the kernel didn't run.

2. **DPRINT on unmonitored cores is safe**: the DPRINT server initializes ALL cores with DISABLED_MAGIC, so unmonitored cores skip DPRINT silently (no buffer overflow deadlock).

3. **Device reset before debugging**: hung processes can leave Metal state in a bad state. Always `tt-smi -r 0,1,2,3,4,5,6,7` before a fresh debugging session if previous runs crashed.

4. **Progress semaphore semantics**: the `t_batches_needed = (t_block - t_out_start) / T_out_block + 1` formula gives t_batches_needed=1 for each core's first T iteration. This works in practice (NP stays ahead of conv3d) but is technically racy for T-parallel cores at non-zero t_out_start. The same behavior exists in the original non-fused path.
