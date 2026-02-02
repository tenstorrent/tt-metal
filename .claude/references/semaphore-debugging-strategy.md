# Semaphore Debugging Strategy

## Overview

This guide provides systematic strategies for debugging semaphore-related synchronization bugs in TTNN kernel code. Semaphores coordinate inter-core and intra-core communication via atomic operations over NoC.

### Common Semaphore Bug Types

| Bug Type | Description | Symptom |
|----------|-------------|---------|
| **Missed signal** | Sender sets wrong value, receiver waits forever | Hang with all cores at `W` |
| **Count mismatch** | Sender expects N receivers, only N-1 signal | Sender hangs at `noc_semaphore_wait` |
| **Race condition** | Multiple `sem_inc` from different cores lose updates | Intermittent hangs |
| **Stale semaphore** | Not reset between loop iterations | Works first iteration, hangs later |
| **Wrong NoC address** | Semaphore address calculated incorrectly | Hang or wrong core signaled |
| **INVALID/VALID confusion** | Using wrong constant values | Immediate or delayed hang |
| **Missing barrier** | `noc_async_atomic_barrier()` missing after `sem_inc` | Race condition |

---

## Strategy 1: Signal-Wait Balance Analysis

### Goal
Verify that the number of signals sent matches the number of signals expected by receivers, and that semaphores are properly reset between iterations.

### Steps

1. **Count signal operations** in sender kernels:
   ```bash
   grep -n "noc_semaphore_inc\|noc_semaphore_set_multicast" sender_kernel.cpp
   ```

2. **Find expected signal threshold** in receiver kernels:
   ```bash
   grep -n "noc_semaphore_wait" receiver_kernel.cpp
   ```
   Look for patterns like:
   ```cpp
   noc_semaphore_wait(sem, N);  // Expects N signals
   ```

3. **Verify receiver count** in program factory:
   ```bash
   grep -n "mcast_num_dests\|mcast_num_cores\|num_receivers" program_factory.cpp
   ```

4. **Check semaphore reset** at loop start:
   ```bash
   grep -B5 -A5 "for.*num_blocks\|while.*iter" kernel.cpp | grep noc_semaphore_set
   ```
   Semaphores should be reset to initial values (often 0 or INVALID) at the beginning of each iteration.

### Common Issues
- Sender waits for `N` receivers but only `N-1` cores are in the receiver range
- Multicast `num_cores` includes/excludes sender incorrectly (off-by-one)
- Semaphore not reset between loop iterations, causing wait to pass immediately on iteration 2+

---

## Strategy 2: DPRINT Counter Instrumentation

### Goal
Track signal and wait counts dynamically to identify where synchronization breaks down.

### Instrumentation Templates

#### Template 1: Sender Signal Tracking
```cpp
// At top of kernel
uint32_t signal_count = 0;

// Before each noc_semaphore_inc or noc_semaphore_set_multicast
signal_count++;
DPRINT << "SIGNAL #" << signal_count << " sent to sem" << sem_id << ENDL();

// After multicast
noc_semaphore_set_multicast(local_sem_addr, remote_noc_addr, num_cores);
DPRINT << "MCAST_SIGNAL to " << num_cores << " cores" << ENDL();
```

#### Template 2: Receiver Wait Tracking
```cpp
// At top of kernel
uint32_t wait_count = 0;

// Before wait
volatile tt_l1_ptr uint32_t* sem = get_semaphore(sem_id);
DPRINT << "WAIT_START sem" << sem_id << " current_val=" << *sem << ENDL();

// After wait returns
noc_semaphore_wait(sem, expected_val);
wait_count++;
DPRINT << "WAIT_DONE #" << wait_count << " sem" << sem_id << " val=" << *sem << ENDL();
```

#### Template 3: Semaphore Value Monitoring
```cpp
volatile tt_l1_ptr uint32_t* sem = get_semaphore(sem_id);
DPRINT << "SEM" << sem_id << " val=" << *sem << " (expect=" << expected_val << ")" << ENDL();
```

#### Template 4: NoC Address Verification
```cpp
uint64_t noc_addr = get_noc_addr(target_x, target_y, sem_l1_addr);
DPRINT << "NOC_ADDR core(" << target_x << "," << target_y << ") addr=" << sem_l1_addr << " noc64=" << noc_addr << ENDL();
```

### Analysis Process
1. Run test with DPRINT enabled
2. Compare signal counts across sender cores
3. Compare wait completions across receiver cores
4. Identify which core(s) never reach `WAIT_DONE`
5. Check if `signal_count * num_senders == wait_threshold * num_receivers`

---

## Strategy 3: Multicast Coordination Analysis

### Goal
Verify sender-receiver multicast patterns match expected topology and synchronization flow.

### Multicast Pattern: Sender-Receiver Coordination

**Expected flow**:
```
Receivers:
  1. Reserve CB space
  2. Set receiver_sem = INVALID (not ready to receive)
  3. Increment sender_sem (signal "I'm ready")
  4. Wait for receiver_sem == VALID (data arrival)
  5. Push CB (data now available)

Sender:
  1. Reserve CB space
  2. Read data from DRAM
  3. Wait for sender_sem == num_receivers (all ready)
  4. Multicast data to receivers
  5. Set receiver_sem = VALID on all receivers (multicast)
  6. Push CB
```

### Verification Steps

1. **Check receiver readiness signaling**:
   ```bash
   grep -A10 "noc_semaphore_inc.*sender_sem" receiver_kernel.cpp
   ```
   Should see pattern: `set INVALID → inc sender_sem → wait VALID`

2. **Check sender wait threshold**:
   ```bash
   grep -n "noc_semaphore_wait.*sender_sem" sender_kernel.cpp
   ```
   Compare threshold value to actual number of receivers

3. **Verify multicast destination count**:
   ```bash
   grep -n "noc_async_write_multicast\|noc_semaphore_set_multicast" sender_kernel.cpp
   ```
   Check `num_cores` parameter matches receiver core range

4. **Check NoC coordinate calculation**:
   ```bash
   grep -B5 "get_noc_addr\|get_noc_multicast_addr" kernel.cpp
   ```
   Verify x, y coordinates match core range from program factory

### Common Issues
- Sender waits for wrong receiver count (e.g., waits for `grid_size` but should wait for `grid_size - 1` if sender is in grid)
- Multicast destination range off by one row/column
- Receiver increments wrong semaphore address (sender never sees signal)
- Forgetting `noc_async_atomic_barrier()` after `noc_semaphore_inc` (race condition)

---

## Strategy 4: Cross-Kernel Semaphore Correlation

### Goal
Match producer signal counts with consumer wait counts across multiple cores to identify imbalances.

### Analysis Steps

1. **Identify semaphore roles** from operation analysis:
   - Which semaphore is for sender → receiver signaling?
   - Which is for receiver → sender readiness?
   - Are semaphores shared across multiple core pairs?

2. **Calculate expected signal totals**:
   ```
   Total signals = signals_per_iteration * num_iterations * num_sender_cores
   ```

3. **Calculate expected wait completions**:
   ```
   Total waits = waits_per_iteration * num_iterations * num_receiver_cores
   ```

4. **Verify balance**:
   - If semaphore is incremented by N producers, consumer should wait for N
   - If semaphore is reset each iteration, verify reset happens before wait

### Multi-Core Aggregation Pattern

**Example**: Barrier synchronization where all cores signal and wait:
```cpp
// Each core does:
noc_semaphore_inc(barrier_sem_noc_addr, 1);
noc_async_atomic_barrier();
noc_semaphore_wait(local_barrier_sem, total_num_cores);
```

**Expected**: Each core's local semaphore value should reach `total_num_cores`

**DPRINT verification**:
```cpp
noc_semaphore_inc(barrier_sem_noc_addr, 1);
noc_async_atomic_barrier();
DPRINT << "BARRIER_SIGNAL sent from core(" << my_x << "," << my_y << ")" << ENDL();

volatile tt_l1_ptr uint32_t* barrier_sem = get_semaphore(barrier_sem_id);
DPRINT << "BARRIER_WAIT start, val=" << *barrier_sem << " (expect=" << total_num_cores << ")" << ENDL();
noc_semaphore_wait(barrier_sem, total_num_cores);
DPRINT << "BARRIER_WAIT done, val=" << *barrier_sem << ENDL();
```

---

## Strategy 5: Systematic Experiment Protocol

### Goal
Use structured hypothesis-testing with DPRINT to isolate root cause.

### Protocol Steps

1. **Form hypothesis** based on watcher and code inspection:
   - Example: "Sender waits for 8 receivers but only 7 are signaling"

2. **Design minimal experiment**:
   - Add DPRINT to count signals sent by receivers
   - Add DPRINT to show sender's wait threshold and current semaphore value

3. **Predict outcome**:
   - If hypothesis correct: DPRINT will show 7 SIGNAL messages, sender stuck at `val=7 (expect=8)`
   - If hypothesis incorrect: DPRINT will show different pattern

4. **Run experiment**:
   ```bash
   pytest <test_command>
   ```

5. **Analyze output**:
   - Count SIGNAL messages: `grep "SIGNAL" test_output.log | wc -l`
   - Find final semaphore value: `grep "WAIT.*sender_sem" test_output.log | tail -1`

6. **Conclude**:
   - If prediction matches: Hypothesis confirmed, propose fix
   - If prediction doesn't match: Refine hypothesis, design new experiment

### Experiment Design Principles
- **Minimize instrumentation**: Only add DPRINT where hypothesis needs verification
- **Use counters**: Track cumulative counts, not just transient states
- **Log before blocking**: Always DPRINT before `wait` to capture values even if test hangs
- **Include context**: Log iteration number, block ID, core coordinates

---

## Quick Reference: DPRINT Templates

### Semaphore Value Check
```cpp
volatile tt_l1_ptr uint32_t* sem = get_semaphore(sem_id);
DPRINT << "SEM" << sem_id << "=" << *sem << ENDL();
```

### Signal Sent (Local Core)
```cpp
noc_semaphore_set(local_sem_addr, value);
DPRINT << "SET sem" << sem_id << "=" << value << ENDL();
```

### Signal Sent (Remote Core)
```cpp
noc_semaphore_inc(remote_sem_noc_addr, increment);
DPRINT << "INC sem to core(" << x << "," << y << ") +=" << increment << ENDL();
```

### Wait Started
```cpp
volatile tt_l1_ptr uint32_t* sem = get_semaphore(sem_id);
DPRINT << "WAIT_START sem" << sem_id << " val=" << *sem << " expect=" << threshold << ENDL();
noc_semaphore_wait(sem, threshold);
```

### Wait Completed
```cpp
noc_semaphore_wait(sem, threshold);
DPRINT << "WAIT_DONE sem" << sem_id << " val=" << *sem << ENDL();
```

### Multicast Sent
```cpp
noc_semaphore_set_multicast(local_sem_addr, remote_noc_addr, num_cores);
DPRINT << "MCAST_SET sem" << sem_id << " to " << num_cores << " cores val=" << value << ENDL();
```

### Iteration/Block Tracking
```cpp
DPRINT << "BLOCK " << block_idx << " of " << num_blocks << ENDL();
```

---

## Decision Tree

Use this tree to guide initial hypothesis formation:

### 1. All cores at `W`, no CB waypoint visible?
→ **Likely semaphore deadlock**
- Check: `grep -n "noc_semaphore_wait" kernel.cpp`
- Hypothesis: Sender/receiver synchronization mismatch

### 2. `NRW` or `NWW` stuck?
→ **NoC stall, could be semaphore operation**
- Check: `grep -n "noc_semaphore_inc\|noc_semaphore_set_multicast" kernel.cpp`
- Hypothesis: NoC congestion or incorrect NoC address

### 3. Sender cores running, receiver cores waiting?
→ **Sender likely stuck at `noc_semaphore_wait`**
- Check sender's wait threshold vs. number of active receivers
- Hypothesis: Receiver count mismatch or receivers not signaling

### 4. Receiver cores running, sender core waiting?
→ **Receivers not signaling sender**
- Check: `grep -n "noc_semaphore_inc.*sender_sem" receiver_kernel.cpp`
- Hypothesis: Receivers missing `noc_semaphore_inc` or using wrong address

### 5. Works first iteration, hangs on subsequent iterations?
→ **Stale semaphore value**
- Check: `grep -B5 "for.*num_blocks" kernel.cpp | grep noc_semaphore_set`
- Hypothesis: Semaphore not reset to initial value at loop start

### 6. Intermittent hangs (not every run)?
→ **Race condition**
- Check: `grep -A1 "noc_semaphore_inc" kernel.cpp | grep noc_async_atomic_barrier`
- Hypothesis: Missing `noc_async_atomic_barrier()` after `noc_semaphore_inc`

### 7. Multicast operation hangs?
→ **Destination count or address mismatch**
- Check: `grep "mcast_num_dests\|mcast_num_cores" program_factory.cpp`
- Compare with: `grep "noc_semaphore_set_multicast.*num_cores" kernel.cpp`
- Hypothesis: Multicast destination count doesn't match semaphore receiver count

---

## Common Pitfalls

### Pitfall 1: Missing noc_async_atomic_barrier()
**Issue**: `noc_semaphore_inc` is an async NoC operation. Without a barrier, the CPU may proceed before the increment reaches the destination.

**Correct pattern**:
```cpp
noc_semaphore_inc(remote_sem_noc_addr, 1);
noc_async_atomic_barrier();  // REQUIRED: Wait for NoC write to complete
```

**Symptom**: Intermittent hangs, works sometimes but not always (race condition)

### Pitfall 2: VALID/INVALID Constant Confusion
**Issue**: Using raw 0/1 instead of defined constants, or swapping VALID and INVALID.

**Correct usage**:
```cpp
constexpr uint32_t INVALID = 0;
constexpr uint32_t VALID = 1;

noc_semaphore_set(receiver_sem, INVALID);  // Mark not ready
noc_semaphore_wait(receiver_sem, VALID);   // Wait for ready
```

**Common mistake**:
```cpp
noc_semaphore_set(receiver_sem, 1);  // Should be INVALID (0)
noc_semaphore_wait(receiver_sem, 0);  // Should be VALID (1)
```

### Pitfall 3: Multicast num_cores Off-By-One
**Issue**: Confusion about whether sender is included in multicast destination count.

**Pattern 1**: Sender multicasts to N receivers (sender NOT in range)
```cpp
// Program factory
uint32_t num_receivers = receiver_cores.size();  // e.g., 7

// Sender kernel
noc_semaphore_wait(sender_sem, num_receivers);  // Wait for 7
noc_semaphore_set_multicast(local, remote, num_receivers);  // Send to 7
```

**Pattern 2**: All cores in grid participate (sender IS in grid)
```cpp
// Program factory
uint32_t grid_size = all_cores.size();  // e.g., 8 (including sender)

// Sender kernel
uint32_t num_dests = grid_size - 1;  // 7 (exclude self)
noc_semaphore_wait(sender_sem, num_dests);  // Wait for 7 others
noc_semaphore_set_multicast(local, remote, num_dests);  // Send to 7
```

### Pitfall 4: Semaphore Not Reset Between Iterations
**Issue**: Semaphore accumulates values across loop iterations, causing waits to pass immediately.

**Incorrect**:
```cpp
for (uint32_t block = 0; block < num_blocks; block++) {
    // Semaphore not reset, still has value from previous iteration
    noc_semaphore_wait(sem, VALID);  // Passes immediately on iteration 2+
}
```

**Correct**:
```cpp
for (uint32_t block = 0; block < num_blocks; block++) {
    noc_semaphore_set(receiver_sem, INVALID);  // Reset at start of iteration
    noc_semaphore_inc(sender_sem_noc_addr, 1);
    noc_semaphore_wait(receiver_sem, VALID);
}
```

### Pitfall 5: Wrong NoC Address Calculation
**Issue**: Using incorrect x, y coordinates or L1 address when calculating NoC addresses.

**Debugging**:
```cpp
uint64_t noc_addr = get_noc_addr(target_x, target_y, sem_l1_addr);
DPRINT << "Signaling core(" << target_x << "," << target_y << ") noc_addr=" << noc_addr << ENDL();

// Verify coordinates match program factory
// Verify sem_l1_addr matches CreateSemaphore allocation
```

**Common mistake**: Using logical core coordinates instead of physical NoC coordinates, or vice versa.

---

## Summary Workflow

When debugging a semaphore hang:

1. **Observe**: Watcher → Identify which cores are stuck at `W` or `NRW`/`NWW`
2. **Map**: Operation analysis → Understand sender/receiver topology and semaphore roles
3. **Hypothesize**: Use decision tree → Form specific hypothesis about signal/wait mismatch
4. **Instrument**: Add targeted DPRINT → Count signals/waits, log semaphore values
5. **Experiment**: Run test → Compare actual counts with expected
6. **Conclude**: Evidence confirms/refutes → Propose fix or refine hypothesis
