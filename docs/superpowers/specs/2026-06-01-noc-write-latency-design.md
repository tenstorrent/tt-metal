# NOC Write Round-Trip Latency Test — Design Spec

**Date:** 2026-06-01
**Branch:** vvukomanovic/quasar-9x4-dm-emu
**Target device:** Quasar 9x4 DM-only emulator

---

## Goal

Add two latency tests to the `data_movement` test suite that measure the round-trip time of a single non-posted NOC write (i.e. time from issuing the write to the barrier completing). A second kernel on the receiver confirms the data arrived on the correct core.

---

## Tests

| Test name | Source core | Dest core | Purpose |
|---|---|---|---|
| `NocWriteLatencyFarCorners` | `(0,0)` | `(8,3)` | Far-corner latency on 9x4 grid |
| `NocWriteLatencyAdjacentCores` | `(0,0)` | `(1,0)` | Adjacent-core latency baseline |

Both tests run on the Quasar 9x4 emulator only and are skipped on all other targets.

---

## Measurement

- **Iterations:** 100 per test
- **Transaction size:** 32 bytes
- **Timing:** `rdcycles()` sampled immediately before and after `noc_async_write_barrier()`
- **Output:** `DEVICE_PRINT` per iteration from sender; one confirmation print from receiver

```
[sender] iter 0: 1234 cycles
[sender] iter 1: 1230 cycles
...
[receiver] core (8,3) received
```

---

## File Layout

```
tests/tt_metal/tt_metal/data_movement/
└── noc_write_latency/
    ├── kernels/
    │   ├── sender.cpp
    │   └── receiver.cpp
    └── test_noc_write_latency.cpp
```

`sources.cmake` gets one new entry:
```cmake
noc_write_latency/test_noc_write_latency.cpp
```

---

## Kernel Design

### sender.cpp

Compile-time args (in order):
| Index | Name | Notes |
|---|---|---|
| 0 | `src_l1_addr` | Physical L1 address for payload |
| 1 | `flag_local_addr` | Physical L1 address for outgoing flag |
| 2 | `dst_l1_data_addr` | Physical L1 address on receiver for payload |
| 3 | `dst_l1_flag_addr` | Physical L1 address on receiver for flag |
| 4 | `dst_noc_x` | NOC X of destination (physical) |
| 5 | `dst_noc_y` | NOC Y of destination (physical) |
| 6 | `num_iterations` | 100 |
| 7 | `transaction_size_bytes` | 32 |

Logic per iteration:
1. Write payload to `src_l1_addr + MEM_L1_UNCACHED_BASE` (bypass cache)
2. Build `dst_data_noc = get_noc_addr(dst_noc_x, dst_noc_y, dst_l1_data_addr)`
3. `noc_async_write(src_l1_addr, dst_data_noc, transaction_size_bytes)`
4. `t0 = rdcycles()`
5. `noc_async_write_barrier()`
6. `t1 = rdcycles()`
7. `DEVICE_PRINT("[sender] iter %u: %u cycles\n", i, t1 - t0)`
8. Write `i + 1` to `flag_local_addr + MEM_L1_UNCACHED_BASE`
9. `noc_async_write(flag_local_addr, dst_flag_noc, sizeof(uint32_t))`
10. `noc_async_write_barrier()`

### receiver.cpp

Compile-time args (in order):
| Index | Name | Notes |
|---|---|---|
| 0 | `flag_l1_addr` | Physical L1 address where flag arrives |
| 1 | `num_iterations` | Must match sender |
| 2 | `my_noc_x` | For the print |
| 3 | `my_noc_y` | For the print |

Logic:
1. Create `volatile tt_l1_ptr uint32_t* flag_ptr = (volatile tt_l1_ptr uint32_t*)(flag_l1_addr + MEM_L1_UNCACHED_BASE)`
2. For each iteration `i`: spin until `*flag_ptr == i + 1`
3. After final iteration: `DEVICE_PRINT("[receiver] core (%u,%u) received\n", my_noc_x, my_noc_y)`

---

## Host Test Design

### Config struct
```cpp
struct NocWriteLatencyConfig {
    CoreCoord src_core;
    CoreCoord dst_core;
    uint32_t num_iterations;
    uint32_t transaction_size_bytes;
};
```

### Helper `run_noc_write_latency(mesh_device, config)`
1. Skip if not Quasar or grid < 9×4
2. Get physical NOC coords for src and dst via `device->worker_core_from_logical_core()`
3. Compute L1 addresses via `get_l1_address_and_size()` from `dm_common`; pick non-overlapping offsets for payload and flag on both cores
4. Create `Program`, add sender kernel on `src_core` and receiver kernel on `dst_core`
5. Wrap in `MeshWorkload`, enqueue, `Finish(cq)`
6. Return `true`

### Tests
```cpp
TEST_F(GenericMeshDeviceFixture, NocWriteLatencyFarCorners) {
    NocWriteLatencyConfig cfg{.src_core={0,0}, .dst_core={8,3},
                               .num_iterations=100, .transaction_size_bytes=32};
    EXPECT_TRUE(run_noc_write_latency(this->mesh_device_, cfg));
}

TEST_F(GenericMeshDeviceFixture, NocWriteLatencyAdjacentCores) {
    NocWriteLatencyConfig cfg{.src_core={0,0}, .dst_core={1,0},
                               .num_iterations=100, .transaction_size_bytes=32};
    EXPECT_TRUE(run_noc_write_latency(this->mesh_device_, cfg));
}
```

---

## Key Quasar L1 Rules (from codebase)

- `MEM_L1_UNCACHED_BASE` is defined in `dev_mem_map.h` as `MEM_L1_BASE + MEM_L1_SIZE`
- **Local writes**: use `addr + MEM_L1_UNCACHED_BASE` to bypass cache
- **NOC source/dest addresses**: pass physical `addr` (no uncached offset) to `get_noc_addr()` and `noc_async_write()`
- **Receiver polling**: cast to `volatile tt_l1_ptr uint32_t*` at `addr + MEM_L1_UNCACHED_BASE`
- Atomics (LR/SC, AMO) do not work on uncached addresses — plain loads/stores only

---

## Out of scope

- Bandwidth sweep (different transaction sizes)
- Multi-iteration flag reset from receiver
- Non-Quasar architectures
