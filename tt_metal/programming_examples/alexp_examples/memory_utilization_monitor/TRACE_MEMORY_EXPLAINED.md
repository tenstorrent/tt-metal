# Understanding Memory During Trace Inference

## The Question

> "When running inference, trace buffers appear in MB and L1 in KB. How can you see real-time changes in memory during inference?"

---

## Why This Happens

### Trace Execution Has Two Phases:

```
┌────────────────────────────────────────────────────────────┐
│ Phase 1: TRACE CAPTURE (First Inference Run)              │
│ ════════════════════════════════════════════════════════   │
│                                                            │
│  1. Allocate TRACE buffer (32-256MB)     ← LARGE!         │
│  2. Allocate L1 data buffers (10-100MB)  ← LARGE!         │
│  3. Load kernels/firmware (5-10MB)       ← Medium          │
│  4. Run inference & record all commands                    │
│  5. Store commands in TRACE buffer                         │
│                                                            │
│  Result: Everything is allocated, commands recorded        │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Phase 2: TRACE EXECUTION (Subsequent Runs)  ← YOU SEE THIS│
│ ═══════════════════════════════════════════════════════    │
│                                                            │
│  1. TRACE buffer: Already allocated ✓                     │
│  2. L1 data buffers: Already allocated ✓                  │
│  3. Kernels/firmware: Already loaded ✓                    │
│  4. Just replay commands (NO NEW ALLOCATIONS!)            │
│                                                            │
│  Result: No memory changes, just execution!                │
└────────────────────────────────────────────────────────────┘
```

### What You're Seeing:

```
During trace execution:
├─ TRACE: 128 MB  ← Pre-allocated command buffer
├─ L1: 11 MB      ← Kernels + firmware + small buffers
└─ DRAM: 0 MB     ← Data might be in DRAM or already in L1

Total: ~139 MB (but looks static!)
```

**The L1 appears small because:**
- Main data buffers were allocated during capture (phase 1)
- During execution (phase 2), only kernel/firmware remains visible as "new" L1
- Large data buffers might be in DRAM, not L1

---

## How to See Real-Time Memory Changes

### Method 1: Monitor During Trace Capture (Recommended)

**Capture phase shows ALL allocations happening:**

```bash
# Terminal 1: Start allocation server
export TT_ALLOC_TRACKING_ENABLED=1
./build/install/bin/allocation_server_poc

# Terminal 2: Run real-time monitor
python3 tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/realtime_monitor.py

# Terminal 3: Run inference (FIRST TIME = capture)
export TT_ALLOC_TRACKING_ENABLED=1
python3 your_inference_script.py --capture-trace
```

**You'll see:**
```
Update #0:   TRACE: 0 MB,    L1: 0 MB
Update #1:   TRACE: 128 MB,  L1: 5 MB    ← Trace buffer allocated!
Update #2:   TRACE: 128 MB,  L1: 45 MB   ← Data buffers allocated!
Update #3:   TRACE: 128 MB,  L1: 48 MB   ← Kernels loaded!
Update #4:   TRACE: 128 MB,  L1: 48 MB   ← Steady state (execution)
```

### Method 2: Monitor During Execution (Limited Info)

**Execution phase shows static memory:**

```bash
# Terminal 2: Monitor
python3 realtime_monitor.py -i 0.1  # Fast updates

# Terminal 3: Run inference (REPLAY = execution)
python3 your_inference_script.py --replay-trace
```

**You'll see:**
```
Update #0:   TRACE: 128 MB,  L1: 48 MB   ← Already allocated
Update #1:   TRACE: 128 MB,  L1: 48 MB   ← No changes (replay)
Update #2:   TRACE: 128 MB,  L1: 48 MB   ← No changes (replay)
...
(Numbers stay the same - this is NORMAL for trace execution!)
```

### Method 3: Use Server Live Output

The allocation server shows **every allocation/deallocation** as it happens:

```bash
# Server output during trace capture:
✓ [PID 12345] Allocated 134217728 bytes of TRACE ...  ← 128MB trace
✓ [PID 12345] Allocated 16777216 bytes of L1 ...      ← 16MB data
✓ [PID 12345] Allocated 8388608 bytes of L1 ...       ← 8MB data
✓ [PID 12345] Allocated 24576 bytes of L1 ...         ← 24KB kernel
... (hundreds of allocations)

# Server output during trace execution:
(silence - no new allocations!)
```

---

## Real-Time Monitor Usage

### Basic Usage:
```bash
# Start monitor (updates every 0.5s)
python3 realtime_monitor.py

# Fast updates (every 0.1s)
python3 realtime_monitor.py -i 0.1

# Show once and exit
python3 realtime_monitor.py --once
```

### Example Output:

```
📊 Memory Usage Monitor (Update #23)
   Timestamp: 2025-10-22 14:30:45

====================================================================================================
Device   Buffers    DRAM            L1              L1_SMALL        TRACE           Total
====================================================================================================
0        346          0.00 MB      11.03 MB        0.50 MB       128.00 MB       139.53 MB
(Δ)      +12          +0.00 MB     +2.35 MB        +0.00 MB        +0.00 MB        +2.35 MB
====================================================================================================

💡 Tips:
   - Δ (delta) shows changes since last update
   - Large TRACE allocations = trace capture phase
   - Small L1 allocations = kernel/firmware loading
   - Steady state = trace execution (no new allocations)
```

---

## Understanding the Numbers

### Typical Memory Profile During Inference:

| Phase | TRACE | L1 (Data) | L1 (Code) | Total | Changes |
|-------|-------|-----------|-----------|-------|---------|
| **Init** | 0 MB | 0 MB | 0 MB | 0 MB | - |
| **Capture Start** | 128 MB | 0 MB | 5 MB | 133 MB | +133 MB |
| **Capture (Data)** | 128 MB | 80 MB | 5 MB | 213 MB | +80 MB |
| **Capture (Kernels)** | 128 MB | 80 MB | 12 MB | 220 MB | +7 MB |
| **Execution** | 128 MB | 80 MB | 12 MB | 220 MB | 0 MB ← Static! |

### Why L1 Appears Small:

1. **L1 ≠ Total Memory**
   - You're seeing L1 (on-chip SRAM) separately from DRAM
   - Large tensors often go to DRAM, not L1
   - Check DRAM column for main data

2. **Kernel/Firmware vs Data**
   - Kernel code: ~5-15MB total (all cores)
   - Data buffers: ~10-200MB (might be in DRAM!)
   - TRACE buffer: ~32-256MB (separate pool)

3. **Already Allocated**
   - During execution, buffers are already allocated
   - No new allocations = no visible changes
   - This is **by design** - trace execution is fast because nothing allocates!

---

## How to Force More L1 Usage (For Testing)

If you want to see larger L1 allocations:

```python
import ttnn
import torch

# Force L1 allocation (not DRAM)
large_tensor = torch.randn(1, 1, 512, 131072, dtype=torch.bfloat16)  # ~128MB
tt_tensor = ttnn.from_torch(
    large_tensor,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,  # ← Force L1!
    dtype=ttnn.bfloat16
)

# Now check realtime monitor - you should see +128MB in L1
```

---

## Summary

### ✅ You CAN See Real-Time Changes:

1. **During trace capture** - lots of allocations
2. **During first inference** - buffers being created
3. **Using realtime_monitor.py** - live updates

### ❌ You WON'T See Changes:

1. **During trace execution** - already allocated
2. **During cached program runs** - buffers reused
3. **If everything is in DRAM** - check DRAM column!

### The Real-Time Monitor Shows:

- Live memory statistics (updates every 0.1-1s)
- Delta (Δ) showing changes since last update
- Per-device breakdown (DRAM, L1, L1_SMALL, TRACE)
- Color-coded changes

### Key Insight:

**"No changes during inference" is NORMAL for trace execution!**
The whole point of traces is to pre-allocate everything once,
then replay commands without any allocation overhead. 🚀

If you want to see dynamic allocations, monitor during:
- Trace capture (first run)
- Model initialization
- Buffer creation/destruction between runs
