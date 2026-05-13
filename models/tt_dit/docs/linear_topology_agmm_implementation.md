# Enabling Linear Topology in `all_gather_minimal_matmul_async`

## Goal

The `all_gather_minimal_matmul_async` op (AGMM) fuses an all-gather of the K dimension with a matmul. It was originally designed and tested only for **Ring** topology. This document records the work to enable **Linear** topology, which is required for inference on Wormhole Galaxy systems where devices are connected in a 1D chain rather than a ring.

Target test:
```bash
scripts/run_safe_pytest.sh \
  models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py \
  -k "wh4x8links4_linear and 4k4k4k and check and fused" -s --tt-arch wormhole_b0
```
Config: M=32768, K=4096, N=4096, ring_size=4, num_links=4, cluster_axis=0, topology=Linear.

---

## Background: The Ring Algorithm

Before describing what was changed, it helps to understand what the existing Ring algorithm does.

Each device holds a "shard" of the K dimension: `K / ring_size` columns of the weight matrix. The all-gather collects all shards so every device can compute a full `M × K` matmul. The two key data movement concepts are:

**Bidirectional half-block relay.** Each device's K-block is split into two halves:
- `k_left` (first half) relays **backward** around the ring (toward lower ring_index, wrapping).
- `k_right` (second half) relays **forward** around the ring (toward higher ring_index, wrapping).

Because both directions travel simultaneously, a device at position `i` receives the full K-block of a remote device `d` in `ring_size/2` hops. This halves the latency vs. unidirectional relay.

**Semaphore handshake.** Relay cores write data into an `ag_output_tensor` buffer and then increment a global semaphore on the destination device. The injector cores (which feed data into the matmul) wait on both the forward and backward semaphores before reading — ensuring the ag_output_tensor is populated before consumption.

**`k_forward` alternation.** The K-block iteration direction flips each N-block pass to optimize memory access patterns.

---

## Why Linear Topology Hangs (Root Causes)

### Root Cause 1 — Semaphore deadlock (primary)

The semaphore wait was unconditional:
```cpp
// matmul_dataflow_common.hpp (original)
noc_semaphore_wait_min(out_ready_semaphore_forward, sem_target_forward + in0_core_order_size);
noc_semaphore_wait_min(out_ready_semaphore_backward, sem_target_backward + in0_core_order_size);
```

For a Linear device at position 0 (`num_targets_backward = 0`): `out_ready_semaphore_backward` is never incremented — there is no ring wrap, so no backward neighbor ever sends to it. The wait blocks forever.

Similarly, device 3 (end of chain) deadlocks on `out_ready_semaphore_forward`.

### Root Cause 2 — Wrong K-tile positions for all devices

The Ring algorithm uses **modular arithmetic** to compute which device's K-block to read at each `device_iter`:
```cpp
// k_left comes from rank + device_iter (mod ring_size)
// k_right comes from rank - device_iter (mod ring_size)
```

For Linear, the modular wrap is incorrect. Device 0 at `device_iter=3` would compute `rank - 3 = -3`, wrapping to `device 1` — but no data from `device 1` arrives via that path.

### Root Cause 3 — Relay of last K-block iterations skipped for intermediate devices

The original relay guard:
```cpp
if (k_block_iter < (K_num_blocks - (K_num_blocks / num_devices))) {
    forward_slice = true;
}
```

This skips relaying the final `K_blocks_per_device` iterations, which is correct for Ring because the ring wrap already delivered those blocks. For Linear, every K-block must be relayed hop-by-hop — there is no wrap. This caused a hang (relay cores on intermediate devices stopped sending before all data reached the far end of the chain).

### Root Cause 4 — `k_forward` alternation breaks monotone ordering

The toggle `k_forward = !k_forward` alternates the direction of K-block iteration each N-block pass. For Linear, the `device_iter`-to-direction mapping (`<= num_targets_fwd` → forward; `> num_targets_fwd` → backward) must be consistent across all N-block passes. Alternating breaks this.

---

## Fix Strategy

Replace the half-block bidirectional Ring scheme with a **full-block unidirectional** scheme for Linear:

- `k_left_tiles = K_block_tiles` (full K-block)
- `k_right_tiles = 0` (no opposing half)
- Data travels in one direction per `device_iter` — no ring wrap
- Semaphore wait is directional: forward device_iters wait on `sem_forward`, backward device_iters wait on `sem_backward`
- `k_forward` is not toggled for Linear — iteration is always monotone ascending

The Ring path is **entirely unchanged** — all new code is behind `if constexpr (is_linear)` guards.

---

## Files Modified

### 1. `kernels/matmul_dataflow_common.hpp`

**What changed:** Converted `compute_actual_k_block` to a template function parameterized by `<HasForwardTargets, HasBackwardTargets, IsLinear>`. Added `num_targets_fwd_rt` parameter (runtime, passed from caller).

**K-tile position fix (both IS_IN0 and non-IS_IN0 paths):**
```cpp
if constexpr (IsLinear) {
    uint32_t actual_device_rank;
    if (device_iter <= num_targets_fwd_rt) {
        actual_device_rank = my_rank + device_iter;  // forward device
    } else {
        actual_device_rank = my_rank - (device_iter - num_targets_fwd_rt);  // backward device
    }
    k_left_start_tile = (actual_device_rank * k_blocks_per_device + device_k_block_iter) * k_tiles_per_block;
    k_right_start_tile = k_left_start_tile;  // unused (k_right_tiles == 0)
}
```

**Semaphore wait fix (IS_IN0 path only):**
```cpp
if constexpr (IsLinear) {
    if (device_iter <= num_targets_fwd_rt) {
        noc_semaphore_wait_min(out_ready_semaphore_forward, sem_target_forward + in0_core_order_size);
        sem_target_forward += in0_core_order_size;
    } else {
        noc_semaphore_wait_min(out_ready_semaphore_backward, sem_target_backward + in0_core_order_size);
        sem_target_backward += in0_core_order_size;
    }
}
```

For Ring: waits on both semaphores (forward and backward), gated by `HasForwardTargets`/`HasBackwardTargets` to prevent deadlock at Ring endpoints.

### 2. `kernels/dm_in0_sender.cpp`

Five changes:

**a. `is_linear` compile-time flag:**
```cpp
constexpr bool is_linear = (topology == Topology::Linear);
```

**b. k_left/k_right tiles (full block for Linear):**
```cpp
if constexpr (is_linear) {
    k_left_tiles = K_block_tiles;
    k_right_tiles = 0;
} else {
    // original half-block split
}
```

**c. `compute_actual_k_block` call — template args and corrected `in0_core_order_size`:**
```cpp
compute_actual_k_block<(num_targets_forward_direction > 0), (num_targets_backward_direction > 0), is_linear>(
    ...,
    is_injector_core,
    in0_core_order_size,   // was hardcoded to 1 (bug — see "Race Condition" below)
    num_targets_forward_direction,
    ...);
```

**d. Relay send logic — direction gating for Linear:**

For Linear, each K-block travels in one direction:
- `dev_iter == 0` (local shard): relay in both directions (first hop of each chain)
- `dev_iter <= num_targets_forward`: relay backward (to device-1)
- `dev_iter > num_targets_forward`: relay forward (to device+1)

Endpoint guards (`if constexpr (num_targets_backward_direction > 0)` / forward) prevent sends beyond chain boundaries, eliminating null-pointer crashes at device 0 and device 3.

For Ring: the original code was refactored behind `} else {` with identical logic plus null-pointer protection via the same constexpr guards.

**e. `k_forward` toggle — disabled for Linear:**
```cpp
if constexpr (!is_linear) {
    k_forward = !k_forward;
}
```

**f. `forward_slice` guard — always true for Linear:**
```cpp
if constexpr (is_linear) {
    forward_slice = true;  // every K-block must be relayed hop-by-hop
} else {
    if (k_block_iter < (K_num_blocks - (K_num_blocks / num_devices))) {
        forward_slice = true;
    }
}
```

### 3. `kernels/dm_in1_sender_out.cpp`

This kernel manages the in1 (weight) tensor and uses `compute_actual_k_block` only for K-tile position calculation (no semaphore logic). Changes mirror dm_in0_sender:

- Added `#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"` and `using ttnn::ccl::Topology`
- Added CT args at indices 20–21: `num_targets_forward_direction`, `topology`
- Fixed `TensorAccessorArgs<20>` → `TensorAccessorArgs<22>` (offset for new CT args)
- Same k_left/k_right topology branch
- `k_forward` toggle guarded by `if constexpr (!is_linear)`

### 4. `all_gather_minimal_matmul_async_program_factory.cpp`

Added `num_targets_forward` and `static_cast<uint32_t>(topology)` to both `in1_sender_compile_time_args` and `in1_receiver_compile_time_args` to supply the new CT args required by dm_in1_sender_out.cpp.

### 5. `all_gather_minimal_matmul_async_nanobind.cpp`

Updated docstring: "Valid options are Ring" → "Valid options are Ring and Linear".

---

## Bugs Encountered and Fixes Applied

### Bug 1: JIT compile error — `TensorAccessorArgs<20>` index out of range

After adding two new CT args to dm_in1_sender_out.cpp at indices 20 and 21, the `TensorAccessorArgs<20>` call shifted to index 22. Fixed by changing to `TensorAccessorArgs<22>`.

### Bug 2: JIT compile error — `expected ';' before numeric constant`

Template arguments `num_targets_forward_direction > 0` were parsed as comparisons in the template call. Fixed by parenthesizing: `(num_targets_forward_direction > 0)`.

### Bug 3: Kernel hang (~14 minutes)

**Symptom:** Test ran without assertion error but never returned.

**Cause:** `forward_slice=false` (the Ring guard) prevented relay of the last `K_blocks_per_device` iterations for intermediate devices. For a 4-device chain with `K_num_blocks=32` and `K_blocks_per_device=8`, this blocked relaying of k_block_iters 24–31. Device 3 never received K-blocks from devices 1 and 2. The injectors on device 3 waited on semaphores that were never incremented.

**Fix:** Set `forward_slice = true` unconditionally for Linear (every K-block must relay until it reaches the far end of the chain).

### Bug 4: PCC ~0.9687 (race condition)

**Symptom:** Test completed, no hang, but `assert 0.9688 > 0.9995` failed.

**Cause:** The `compute_actual_k_block` call in dm_in0_sender.cpp had a hardcoded `1` for the `in0_core_order_size` parameter (line 351). This parameter is used inside the function as the semaphore threshold increment:
```cpp
noc_semaphore_wait_min(sem, sem_target + in0_core_order_size);
```

The actual value should be 8 (= `in0_core_order.size()` = `in1_parallel_axis_cores` = `grid_size.y` with `force_transpose=True`). There are 8 relay cores per direction (one per x-column). Each increments the destination device's semaphore by 1 when it finishes writing its M-row range. The injector must wait for all 8 increments before reading.

With `1` hardcoded, the semaphore wait was satisfied after the first relay core finished — 7 out of 8 M-row ranges could be unwritten. This created a race condition that corrupted ~3% of the output tiles, yielding PCC ~0.969.

**Fix:** Changed `1` to `in0_core_order_size` (the local variable read from kernel args at line 132).

---

## Current State

The PCC fix (Bug 4) has been applied and the code rebuilt. The test has not been confirmed passing because the hardware entered a bad state (devices became undetectable by `tt-smi`) during the debugging session. Specifically:

1. A kernel hang triggered an interruption of the test process
2. Manual `tt-smi -r` was called but the CPLD FW version on this Galaxy system requires `tt-smi -glx_reset`
3. `tt-smi -glx_reset` reset the trays but failed at re-initialization: `ioctl get_device_info failed for device 0 with: ENODEV`
4. Subsequent `tt-smi -r` and `tt-smi -glx_reset` calls all fail — the driver can no longer see the devices

Device recovery requires root-level PCIe rescan or driver reload (not available in this environment), or a host reboot.

---

## Next Steps

1. **Recover hardware** — reboot or PCIe rescan with root privileges, or `sudo modprobe -r tenstorrent && sudo modprobe tenstorrent`.

2. **Run Linear test** — confirm PCC fix passes:
   ```bash
   scripts/run_safe_pytest.sh \
     models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py \
     -k "wh4x8links4_linear and 4k4k4k and check and fused" -s --tt-arch wormhole_b0
   ```

3. **Run Ring regression** — confirm existing Ring behavior is unaffected:
   ```bash
   scripts/run_safe_pytest.sh \
     models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py \
     -k "wh4x8links4_ring and check" -s --tt-arch wormhole_b0
   ```

4. **Broader regression** — all wh4x8 check configs:
   ```bash
   scripts/run_safe_pytest.sh \
     models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py \
     -k "wh4x8 and check" -s --tt-arch wormhole_b0
   ```

5. **Open question: relay direction logic for intermediate devices.** The current relay logic in dm_in0_sender.cpp for Linear uses `dev_iter` to decide which direction to relay. This was written to match the forward/backward device ordering in `compute_actual_k_block`. It has not been fully validated on hardware for the PCC-correct case. If the PCC fix (Bug 4) alone does not fully resolve the issue after hardware recovery, the next place to look is the relay direction logic for intermediate (non-endpoint) devices in the `forward_slice` section.

---

## Key Invariants

- **Ring path untouched**: all new logic is behind `if constexpr (is_linear)` — existing Ring tests must continue to pass without any behavior change.
- **Endpoint safety**: `if constexpr (num_targets_backward_direction > 0)` / forward guards prevent null pointer dereferences at chain endpoints (device 0 has no backward neighbor, device N-1 has no forward neighbor).
- **Direction naming**: `mux_connection_handle_forward` sends to `device-1` (backward_coord); `mux_connection_handle_backward` sends to `device+1` (forward_coord). The names are counterintuitive — do not reverse them.
- **Semaphore naming**: `out_ready_sem_backward` (semaphore[0]) is incremented by the relay that sends toward `device+1`; `out_ready_sem_forward` (semaphore[1]) is incremented by the relay toward `device-1`.
