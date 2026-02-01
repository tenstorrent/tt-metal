# Non-Causal SDPA Data Movement Optimization Plan

## Executive Summary

Back-port the KV store-and-forward chain optimization from RingAttention (PR #34929) to standard non-causal SDPA to reduce duplicate DRAM accesses and achieve 75% math utilization.

## Problem Statement

### Current Issue
SDPA partitions work across cores based on Q chunks. For a given configuration:
- Total Q chunks = `num_heads * (seq_len / Q_chunk_size)`
- Q chunks for the same head are spread across multiple cores
- **All cores processing Q chunks for the same head must read the same K and V chunks**
- This leads to **duplicate DRAM accesses**, making the operation data movement-bound

### Impact
- Operations become DM-bound instead of compute-bound
- Math utilization significantly below the 75% target
- RingAttention saw improvement from 44.5ms to 27.4ms (~42% utilization) with this fix

## Solution Overview

Implement a **store-and-forward chain** mechanism where:
1. Cores processing Q chunks for the same head form a "chain"
2. The first core (injector) reads KV chunks from DRAM
3. Subsequent cores receive KV chunks via L1-to-L1 transfer from the previous core
4. Each core forwards to the next until reaching the last core (sink)

This reduces DRAM reads from `N * num_KV_chunks` to just `num_KV_chunks` per head, where N is the number of cores processing that head.

---

## Development Environment Notes

### Build Requirements

| Component | Build Required | Command |
|-----------|---------------|---------|
| **Kernel code** (`.cpp` in `kernels/`) | **NO** - JIT compiled at runtime | N/A |
| **Host C++ code** (program factories, etc.) | **YES** | `./build_metal.sh --release` |
| **Python test code** | **NO** | N/A |

### Handling Device Hangs

During development iteration, tests may hang due to semaphore deadlocks or other synchronization issues.

**Recovery procedure:**
```bash
# Reset the device when a hang is observed
tt-smi -r

# Then re-run the test
```

### Test Execution Strategy

- **Use tight timeouts (5 seconds)** during iteration to quickly detect hangs
- **Build small, fast unit tests** that execute in <1 second when passing
- Run with: `pytest test_file.py -v --timeout=5`

---

## Detailed Implementation Plan

### Phase 1: Build Unit Tests (Test-First Approach)

**Goal:** Create minimal unit tests that will validate the optimization, and verify they pass with the current (non-optimized) implementation.

#### 1.1 Create Test File

**Location:** `tests/ttnn/unit_tests/operations/transformer/test_sdpa_kv_chain_forward.py`

```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.utility_functions import comp_pcc


def run_sdpa_test(device, B, NH, S, DH, dtype=ttnn.bfloat16):
    """
    Run non-causal SDPA and compare against PyTorch reference.

    Uses small shapes to ensure fast execution (<1s).
    """
    torch.manual_seed(42)

    # Create random inputs
    q = torch.randn(B, NH, S, DH)
    k = torch.randn(B, NH, S, DH)
    v = torch.randn(B, NH, S, DH)

    # PyTorch reference (non-causal)
    scale = 1.0 / (DH ** 0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    expected = torch.matmul(attn, v)

    # TTNN execution
    q_tt = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    k_tt = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    v_tt = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Non-causal SDPA
    output_tt = ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_tt, v_tt,
        is_causal=False,
    )

    output = ttnn.to_torch(output_tt)

    # Verify correctness
    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    return passing, pcc


class TestSDPAKVChainForward:
    """
    Test suite for SDPA KV chain forwarding optimization.

    All tests use small shapes for fast iteration (<1s execution).
    """

    @pytest.mark.parametrize(
        "B, NH, S, DH",
        [
            # Minimal test - single head, fits on few cores
            (1, 1, 64, 64),
            # Multi-head - forces Q chunks across cores (triggers chain)
            (1, 8, 128, 64),
            # Larger sequence - more Q chunks per head
            (1, 4, 256, 64),
            # Multi-batch
            (2, 4, 128, 64),
        ],
        ids=[
            "minimal_1h",
            "multi_head_8h",
            "longer_seq_256",
            "multi_batch",
        ]
    )
    def test_sdpa_non_causal_correctness(self, device, B, NH, S, DH):
        """Test non-causal SDPA produces correct results."""
        passing, pcc = run_sdpa_test(device, B, NH, S, DH)
        assert passing, f"PCC check failed: {pcc}"

    def test_sdpa_chain_trigger_shape(self, device):
        """
        Test with shape that guarantees chain forwarding is triggered.

        With NH=8, S=256, chunk_size=32:
        - Q chunks per head = 256/32 = 8
        - Total Q chunks = 8 * 8 = 64
        - With 64 cores, each head's Q chunks span multiple cores
        """
        B, NH, S, DH = 1, 8, 256, 64
        passing, pcc = run_sdpa_test(device, B, NH, S, DH)
        assert passing, f"PCC check failed: {pcc}"
```

#### 1.2 Verify Baseline Tests Pass

**Before any implementation changes**, run the tests to establish baseline:

```bash
# Run with tight timeout to detect any existing issues
pytest tests/ttnn/unit_tests/operations/transformer/test_sdpa_kv_chain_forward.py -v --timeout=5

# If tests hang, reset device and investigate:
tt-smi -r
```

**Expected result:** All tests should PASS with current implementation.

#### 1.3 Test Shape Rationale

| Test | Shape | Why It Matters |
|------|-------|----------------|
| `minimal_1h` | B=1, NH=1, S=64 | Baseline sanity check |
| `multi_head_8h` | B=1, NH=8, S=128 | Multiple heads = Q chunks spread across cores |
| `longer_seq_256` | B=1, NH=4, S=256 | More K chunks to read/forward |
| `multi_batch` | B=2, NH=4, S=128 | Verify batch dimension handling |
| `chain_trigger` | B=1, NH=8, S=256 | Maximally exercises chain forwarding |

---

### Phase 2: Host-Side Implementation (sdpa_program_factory.cpp)

**After tests pass on baseline, implement host-side changes.**

#### 2.1 Add Data Structures for Chain Management

**Location:** `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`

Add the following structures (adapted from `ring_joint_sdpa_program_factory.cpp:671-702`):

```cpp
struct CoreHeadWork {
    uint32_t batch = 0;
    uint32_t head = 0;
    uint32_t q_chunk_start = 0;
    uint32_t q_chunk_count = 0;
};

struct CoreWork {
    CoreCoord logical_core;
    CoreCoord physical_core;
    uint32_t global_q_start = 0;
    uint32_t global_q_count = 0;
    std::vector<CoreHeadWork> head_work;
};

struct HeadSegmentRef {
    uint32_t core_idx = 0;
    uint32_t head_work_index = 0;
};

struct CoreChainInfo {
    bool participates = false;
    bool is_injector = false;
    bool is_sink = false;
    uint32_t batch = 0;
    uint32_t head = 0;
    uint32_t q_chunk_start = 0;
    uint32_t q_chunk_count = 0;
    CoreCoord prev_physical = CoreCoord{0, 0};
    CoreCoord next_physical = CoreCoord{0, 0};
    uint32_t next_core_q_chunks = 0;
};
```

#### 2.2 Create Semaphores for L1-L1 Communication

**Location:** After circular buffer creation in `sdpa_program_factory.cpp`

```cpp
// Only create semaphores for non-causal path
if (!is_causal) {
    auto sender_semaphore_id = CreateSemaphore(program, core_grid, INVALID);
    auto receiver_semaphore_id = CreateSemaphore(program, core_grid, INVALID);
    auto valid_semaphore_id = CreateSemaphore(program, core_grid, VALID);

    reader_compile_time_args.push_back(sender_semaphore_id);
    reader_compile_time_args.push_back(receiver_semaphore_id);
    reader_compile_time_args.push_back(valid_semaphore_id);
}
```

#### 2.3 Implement Chain Construction Algorithm

**Reference:** `ring_joint_sdpa_program_factory.cpp:704-822`

Key steps:
1. Compute flat distribution of Q chunks across cores
2. Track which (batch, head) combinations each core handles
3. For heads spanning multiple cores, build chain:
   - Find first core with single head segment as injector candidate
   - Link cores in sequence: injector -> middle nodes -> sink
   - Store prev/next physical coordinates for NOC transfers

```cpp
// Pseudo-code for chain construction
for (auto& segments : head_segments) {
    if (segments.size() < 2) continue;  // No chain needed

    // Find valid chain start (core handling only one head segment)
    std::optional<size_t> chain_start_idx;
    for (size_t idx = 0; idx + 1 < segments.size(); ++idx) {
        if (core_work[segments[idx].core_idx].head_work.size() == 1) {
            chain_start_idx = idx;
            break;
        }
    }

    if (!chain_start_idx) continue;

    // Build chain from start to end
    for (size_t idx = chain_start_idx.value(); idx < segments.size(); ++idx) {
        // Set chain info: participates, is_injector, is_sink
        // Set prev_physical and next_physical coordinates
        // Set next_core_q_chunks for forwarding count
    }
}
```

#### 2.4 Update Runtime Args

Add chain metadata to reader runtime args:
```cpp
reader_args.push_back(static_cast<uint32_t>(chain.participates));
reader_args.push_back(static_cast<uint32_t>(chain.is_injector));
reader_args.push_back(static_cast<uint32_t>(chain.is_sink));
reader_args.push_back(chain.batch);
reader_args.push_back(chain.head);
reader_args.push_back(chain.q_chunk_start);
reader_args.push_back(chain.q_chunk_count);
reader_args.push_back(static_cast<uint32_t>(chain.prev_physical.x));
reader_args.push_back(static_cast<uint32_t>(chain.prev_physical.y));
reader_args.push_back(static_cast<uint32_t>(chain.next_physical.x));
reader_args.push_back(static_cast<uint32_t>(chain.next_physical.y));
reader_args.push_back(chain.next_core_q_chunks);
```

#### 2.5 Build and Test

```bash
# Build host code changes
./build_metal.sh --release

# Run tests with tight timeout
pytest tests/ttnn/unit_tests/operations/transformer/test_sdpa_kv_chain_forward.py -v --timeout=5

# If hang detected:
tt-smi -r
```

---

### Phase 3: Kernel-Side Implementation (reader_interleaved.cpp)

**Note:** Kernel code is JIT compiled - no build step required. Changes take effect on next test run.

#### 3.1 Add Chain Parameters to Kernel

**Location:** `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`

Add compile-time args for semaphore IDs (only for non-causal):
```cpp
#if !is_causal
uint32_t sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(SENDER_SEM_IDX));
uint32_t receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(RECEIVER_SEM_IDX));
uint32_t valid_semaphore_addr = get_semaphore(get_compile_time_arg_val(VALID_SEM_IDX));
#endif
```

Add runtime args parsing:
```cpp
const uint32_t is_chain_participant = get_arg_val<uint32_t>(argidx++);
const uint32_t is_injector = get_arg_val<uint32_t>(argidx++);
const uint32_t is_sink = get_arg_val<uint32_t>(argidx++);
const uint32_t chain_batch = get_arg_val<uint32_t>(argidx++);
const uint32_t chain_head = get_arg_val<uint32_t>(argidx++);
const uint32_t chain_q_chunk_start = get_arg_val<uint32_t>(argidx++);
const uint32_t chain_q_chunk_count = get_arg_val<uint32_t>(argidx++);
const uint32_t prev_physical_x = get_arg_val<uint32_t>(argidx++);
const uint32_t prev_physical_y = get_arg_val<uint32_t>(argidx++);
const uint32_t next_physical_x = get_arg_val<uint32_t>(argidx++);
const uint32_t next_physical_y = get_arg_val<uint32_t>(argidx++);
const uint32_t next_core_q_chunks = get_arg_val<uint32_t>(argidx++);
```

#### 3.2 Initialize Semaphore Pointers and NOC Addresses

```cpp
volatile tt_l1_ptr uint32_t* valid_semaphore_addr_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_semaphore_addr);
*(valid_semaphore_addr_ptr) = VALID;

volatile tt_l1_ptr uint32_t* receiver_semaphore_addr_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);
volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);

const uint64_t sender_semaphore_noc_addr = get_noc_addr(prev_physical_x, prev_physical_y, sender_semaphore_addr);
const uint64_t receiver_semaphore_noc_addr = get_noc_addr(next_physical_x, next_physical_y, receiver_semaphore_addr);
```

#### 3.3 Modify K Chunk Read Logic

**Reference:** `ring_joint_reader.cpp:222-252`

Replace direct DRAM read with conditional chain logic:

```cpp
// K: either read locally (injector or not participant) or receive from previous core
cb_reserve_back(cb_k_in, k_chunk_tiles);
uint32_t cb_k_start_address = get_write_ptr(cb_k_in);

if (is_injector || !is_chain_participant || (nb != chain_batch || nq != chain_head)) {
    // Original DRAM read path
    read_chunk_with_padding<k_tile_bytes>(
        k_reader, cb_k_in, k_start_tile_id,
        k_row_tile_count, DHt, Sk_chunk_t, DHt,
        barrier_threshold, true /*transpose*/
    );
} else {
    // Receive forwarded K chunk from previous core
    noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);
    noc_semaphore_inc(sender_semaphore_noc_addr, 1);
    noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);
    cb_push_back(cb_k_in, k_chunk_tiles);
}

// Forward K chunk to next core if applicable
if (is_chain_participant && !is_sink && (nb == chain_batch && nq == chain_head) &&
    (q_iter_local < next_core_q_chunks)) {
    noc_semaphore_wait(sender_semaphore_addr_ptr, 1);
    noc_semaphore_set(sender_semaphore_addr_ptr, 0);
    uint64_t k_unicast_data_addr = get_noc_addr(next_physical_x, next_physical_y, cb_k_start_address);
    noc_async_write(cb_k_start_address, k_unicast_data_addr, k_chunk_tiles * k_tile_bytes);
    noc_async_writes_flushed();
    noc_semaphore_set_remote(valid_semaphore_addr, receiver_semaphore_noc_addr);
}
```

#### 3.4 Modify V Chunk Read Logic

Apply identical pattern for V chunks:

```cpp
// V: either read locally (injector or not participant) or receive from previous core
cb_reserve_back(cb_v_in, v_chunk_tiles);
uint32_t cb_v_start_address = get_write_ptr(cb_v_in);

if (is_injector || !is_chain_participant || (nb != chain_batch || nq != chain_head)) {
    // Original DRAM read path
    read_chunk_with_padding<v_tile_bytes>(
        v_reader, cb_v_in, k_start_tile_id,
        k_row_tile_count, vDHt, Sk_chunk_t, vDHt,
        barrier_threshold, false /*no transpose*/
    );
} else {
    // Receive forwarded V chunk from previous core
    noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);
    noc_semaphore_inc(sender_semaphore_noc_addr, 1);
    noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);
    cb_push_back(cb_v_in, v_chunk_tiles);
}

// Forward V chunk to next core if applicable
if (is_chain_participant && !is_sink && (nb == chain_batch && nq == chain_head) &&
    (q_iter_local < next_core_q_chunks)) {
    noc_semaphore_wait(sender_semaphore_addr_ptr, 1);
    noc_semaphore_set(sender_semaphore_addr_ptr, 0);
    uint64_t v_unicast_data_addr = get_noc_addr(next_physical_x, next_physical_y, cb_v_start_address);
    noc_async_write(cb_v_start_address, v_unicast_data_addr, v_chunk_tiles * v_tile_bytes);
    noc_async_writes_flushed();
    noc_semaphore_set_remote(valid_semaphore_addr, receiver_semaphore_noc_addr);
}
```

#### 3.5 Test Kernel Changes

```bash
# No build needed for kernel changes - JIT compiled

# Run tests with tight timeout
pytest tests/ttnn/unit_tests/operations/transformer/test_sdpa_kv_chain_forward.py -v --timeout=5

# If hang detected:
tt-smi -r
```

---

### Phase 4: Iteration Loop

Repeat until all tests pass:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Make code change (kernel or host)                       │
│  2. If host change: ./build_metal.sh --release              │
│  3. Run: pytest ... -v --timeout=5                          │
│  4. If HANG: tt-smi -r, debug semaphore logic               │
│  5. If FAIL: check PCC, debug computation                   │
│  6. If PASS: move to next test case                         │
└─────────────────────────────────────────────────────────────┘
```

### Common Hang Causes and Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Immediate hang | Semaphore never signaled | Check is_injector logic |
| Hang on 2nd K chunk | Forwarding count mismatch | Check next_core_q_chunks |
| Hang on specific head | Chain not built for head | Check head_segments logic |
| Random hangs | Race condition | Add noc_async_writes_flushed() |

---

### Phase 5: Configuration and Feature Flag

#### 5.1 Add Program Config Option

**Location:** `ttnn/cpp/ttnn/operations/transformer/sdpa_config.hpp`

```cpp
struct SDPAProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t q_chunk_size = 32;
    std::size_t k_chunk_size = 32;
    std::optional<bool> exp_approx_mode;
    uint32_t max_cores_per_head_batch = 16;
    bool enable_kv_chain_forwarding = true;  // NEW: Enable store-and-forward
};
```

#### 5.2 Conditional Enablement

In `sdpa_program_factory.cpp`, only enable chain optimization when:
- `!is_causal` (non-causal attention)
- `enable_kv_chain_forwarding` is true
- Head spans multiple cores (optimization provides benefit)

---

### Phase 6: Verify All Tests Pass

Once implementation is complete, run all unit tests multiple times to verify stability:

```bash
# Run full test suite 10 times to check for flaky behavior
for i in {1..10}; do
    echo "=== Run $i ==="
    pytest tests/ttnn/unit_tests/operations/transformer/test_sdpa_kv_chain_forward.py -v --timeout=5
    if [ $? -ne 0 ]; then
        echo "FAILED on run $i"
        tt-smi -r
        exit 1
    fi
done
echo "All runs passed!"
```

---

## File Modification Summary

| File | Changes | Build Required |
|------|---------|----------------|
| `test_sdpa_kv_chain_forward.py` | NEW: Unit tests | No |
| `sdpa_program_factory.cpp` | Chain construction, semaphores, runtime args | Yes: `./build_metal.sh --release` |
| `reader_interleaved.cpp` | Chain receive/forward logic for K and V | No (JIT) |
| `sdpa_config.hpp` | Add `enable_kv_chain_forwarding` option | Yes |

---

## Quick Reference Commands

```bash
# Build host code after C++ changes
./build_metal.sh --release

# Run all unit tests with tight timeout
pytest tests/ttnn/unit_tests/operations/transformer/test_sdpa_kv_chain_forward.py -v --timeout=5

# Run single parametrized test
pytest tests/ttnn/unit_tests/operations/transformer/test_sdpa_kv_chain_forward.py::TestSDPAKVChainForward::test_sdpa_non_causal_correctness[minimal_1h] -v --timeout=5

# Reset device after hang
tt-smi -r

# Check device status
tt-smi
```

---

## Success Criteria

1. All unit tests pass with `--timeout=5`
2. PCC >= 0.99 against PyTorch reference
3. No hangs (run 10x to verify stability)

---

## References

- PR #34929: RingAttention data movement optimization
- `ring_joint_sdpa_program_factory.cpp`: Chain construction reference
- `ring_joint_reader.cpp`: Store-and-forward kernel reference
- Issue #33893: Original RingAttention optimization ticket
