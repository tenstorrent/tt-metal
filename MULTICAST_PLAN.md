# Replace Store-and-Forward with Multicast in minimal_matmul

## Status

Kernel and host factory changes are implemented. **Needs build + test on `bh-glx-c03u08`.**

## Context

`minimal_matmul` distributes data across a core grid using a store-and-forward chain:
- **in0** (activation, M×K): injector core at `in1_idx==0` reads from DRAM, unicasts to the next core, which unicasts to the next, etc. — a serial chain along the x-axis (non-transposed) or y-axis (transposed).
- **in1** (weight, K×N): same pattern along the perpendicular axis.

Each hop in the chain requires a round-trip semaphore handshake. For an 8×8 grid, in0 has 7 serial hops. Replacing this with a single multicast write should reduce distribution latency from O(N) to O(1).

---

## Changes Implemented

### `dm_in0_sender.cpp`

- Added CTA 22: `in0_mcast_num_dests` (injectors: `in1_parallel_axis_cores - 1`; receivers: `0`)
- Shifted `TensorAccessorArgs<22>` → `TensorAccessorArgs<23>` (all downstream offsets auto-adjust)
- Replaced runtime args:
  - Removed `is_sink_core`
  - Replaced `[in0_dest_noc_x, in0_dest_noc_y]` with `[in0_mcast_start_x, in0_mcast_start_y, in0_mcast_end_x, in0_mcast_end_y]`
  - `in0_sender_noc_x/y` still present but now receivers point to the **injector** (not previous hop)
- Removed `in0_receiver_semaphore_noc_addr` (unicast); added `in0_mcast_sem_noc_addr` (precomputed multicast semaphore address)
- Replaced `if (!is_sink_core)` unicast block with:
  ```cpp
  if constexpr (is_injector_core) {
      if constexpr (in0_mcast_num_dests > 0) {
          noc_semaphore_wait(sender_sem_ptr, in0_mcast_num_dests);  // wait for all receivers ready
          noc_semaphore_set(sender_sem_ptr, 0);
          noc_async_write_multicast(src, mcast_data_addr, bytes, in0_mcast_num_dests);
          #ifdef ARCH_BLACKHOLE
          noc_async_writes_flushed();
          #endif
          noc_semaphore_set_multicast(in0_valid_semaphore_addr, in0_mcast_sem_noc_addr, in0_mcast_num_dests);
      }
  }
  ```
- Receiver path unchanged in structure (`cb_reserve_back` → signal injector → wait on local semaphore)

### `dm_in1_sender_out.cpp`

Same changes as above, applied to in1:
- Added CTA 21: `in1_mcast_num_dests`
- Shifted `TensorAccessorArgs<21>` → `TensorAccessorArgs<22>`
- Same runtime arg restructuring
- Row-by-row multicast loop (matching original row-by-row unicast for N-padding correctness)

### `minimal_matmul_program_factory.cpp`

- Removed `build_core_order_for_axis`, `clamped_prev`, `clamped_next` (no longer needed)
- Added `in{0,1}_mcast_num_dests` to sender CTA vectors; `0` to receiver CTA vectors
- Per-core loop now computes:
  - Injector logical coord and physical coord
  - Mcast bounding box (logical → physical, min/max ordered)
- New unified runtime arg layout (same slot count for injectors and receivers):
  - **Injectors**: real mcast start/end physical coords; zero sender noc
  - **Receivers**: zero mcast coords; injector physical NOC as sender noc
- Updated `override_runtime_arguments` index constants (+1 shift for in0 ternary/out indices, same for in1)

**New runtime arg layouts:**

| Index | in0 | in1 |
|-------|-----|-----|
| 0 | in0_addr | in1_addr |
| 1 | in2_addr (bias) | in2_addr (bias) |
| 2 | in3_addr (ag input) | mcast_start_x |
| 3 | mcast_start_x | mcast_start_y |
| 4 | mcast_start_y | mcast_end_x |
| 5 | mcast_end_x | mcast_end_y |
| 6 | mcast_end_y | sender_noc_x |
| 7 | sender_noc_x | sender_noc_y |
| 8 | sender_noc_y | M_start_tile |
| 9 | M_start_tile | M_end_tile |
| 10 | M_end_tile | N_start_tile |
| 11 | N_start_tile | N_end_tile |
| 12 | N_end_tile | defer_write_k_block |
| 13 | defer_write_k_block | [ternary_a_addr] |
| 14 | [ternary_a_addr] | [ternary_b_addr] |
| 15 | [ternary_b_addr] | [out_addrs...] |
| 16+ | [out_addrs...] | |

### `test_minimal_matmul.py`

Added `test_perf_mcast_configs` — sweeps 1k/2k/4k square matmuls at HiFi2/bf16/fp32_acc, 3 warmup + 5 timed runs, reports TFLOPS, checks PCC > 0.9995.

---

## How to Test

```bash
ssh bh-glx-c03u08
cd /data/cglagovich/tt-metal/.claude/worktrees/cglagovich/mm_mcast
git checkout cglagovich/mm_mcast_multicast

# Build
./build_metal.sh

source /data/cglagovich/metal_env.sh && source python_env/bin/activate

# Accuracy tests (should all pass)
pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py \
    -k "not skip and not performance" -v

# Perf test (reports TFLOPS for 1k/2k/4k)
pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py::test_perf_mcast_configs -v -s
```

If the device hangs: `tt-smi -glx_reset && sleep 10`

---

## Key Invariant

The CB write pointer must be identical on the injector and all receivers when the multicast fires.
This is guaranteed because:
1. All cores start with the same CB state
2. Receivers call `cb_reserve_back` **before** signaling readiness (so their write ptr is advanced)
3. The injector waits for all receivers to be ready before firing the multicast
4. All cores advance their CBs in lockstep (same number of `cb_reserve_back` / `cb_push_back` calls per iteration)
