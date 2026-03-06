# GPT-OSS MoE MLP: Parallelism and Communication Diagrams

## 4×8 Galaxy Mesh Layout

### Physical Device Topology
```
                    Tensor Parallel Dimension (TP=8)
                    Columns: Weight Sharding
         ┌────┬────┬────┬────┬────┬────┬────┬────┐
         │ C0 │ C1 │ C2 │ C3 │ C4 │ C5 │ C6 │ C7 │
    ┌────┼────┼────┼────┼────┼────┼────┼────┼────┤
  R │ R0 │ D0 │ D1 │ D2 │ D3 │ D4 │ D5 │ D6 │ D7 │
  o │    ├────┼────┼────┼────┼────┼────┼────┼────┤
  w │ R1 │ D8 │ D9 │D10 │D11 │D12 │D13 │D14 │D15 │
  s │    ├────┼────┼────┼────┼────┼────┼────┼────┤
    │ R2 │D16 │D17 │D18 │D19 │D20 │D21 │D22 │D23 │
  E │    ├────┼────┼────┼────┼────┼────┼────┼────┤
  P │ R3 │D24 │D25 │D26 │D27 │D28 │D29 │D30 │D31 │
  = └────┴────┴────┴────┴────┴────┴────┴────┴────┘
  4
    Expert Parallel Dimension (EP=4)
    Rows: Expert Distribution
```

### Logical Expert Distribution
```
Row 0 (Devices 0-7):   Experts 0-31
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│E0│E1│E2│E3│E4│E5│...│E28│E29│E30│E31│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
 D0:4 experts D1:4 experts ... D7:4 experts

Row 1 (Devices 8-15):  Experts 32-63
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│E32│E33│...                  ...│E63│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Row 2 (Devices 16-23): Experts 64-95
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│E64│E65│...                  ...│E95│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Row 3 (Devices 24-31): Experts 96-127
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│E96│E97│...                 ...│E127│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

## Token Flow with All-to-All Communication

### Initial Token Distribution
```
Batch of 128 Tokens
┌────────────────────────────────────┐
│ Tokens 0-127 (Sequential)          │
└────────────────────────────────────┘
           ↓ Row Sharding
┌─────────┬─────────┬─────────┬─────────┐
│  Row 0  │  Row 1  │  Row 2  │  Row 3  │
│ Tok 0-31│Tok 32-63│Tok 64-95│Tok 96+ │
└─────────┴─────────┴─────────┴─────────┘
```

### All-to-All Dispatch Pattern
```
Before All-to-All: Tokens with Expert Assignments
Row 0: Token 0 →[E5,E42,E67,E89,E102,E15,E28,E99]
       Token 1 →[E12,E55,E78,E91,E8,E33,E120,E45]
       ...
       Token 31→[Expert assignments...]

After All-to-All: Tokens Grouped by Expert Location
Row 0: [Tokens for E0-E31]
       E0: [T15, T89, ...]
       E1: [T3, T45, T112, ...]
       ...
       E31: [T7, T28, ...]

Row 1: [Tokens for E32-E63]
Row 2: [Tokens for E64-E95]
Row 3: [Tokens for E96-E127]
```

### All-to-All Ring Communication
```
Step 1: Initial State
┌────┐    ┌────┐    ┌────┐    ┌────┐
│ R0 │────│ R1 │────│ R2 │────│ R3 │
│ T₀ │    │ T₁ │    │ T₂ │    │ T₃ │
└────┘    └────┘    └────┘    └────┘
  ↑                              ↓
  └──────────────────────────────┘

Step 2: First Exchange (Send to neighbor)
R0 sends tokens for R1's experts → R1
R1 sends tokens for R2's experts → R2
R2 sends tokens for R3's experts → R3
R3 sends tokens for R0's experts → R0

Step 3: Second Exchange
R0 sends tokens for R2's experts → R2 (via R1)
R1 sends tokens for R3's experts → R3 (via R2)
...

Step 4: Third Exchange
R0 sends tokens for R3's experts → R3 (via R2,R1)
...

Result: All tokens arrive at expert devices
```

## Tensor Parallelism Weight Sharding

### MLP Weight Distribution
```
Gate/Up Weight Matrix: [2880, 12288]
                          ↓
        Column-wise Sharding (TP=8)
┌──────┬──────┬──────┬─────┬──────┬──────┬──────┬──────┐
│Col 0 │Col 1 │Col 2 │ ... │Col 5 │Col 6 │Col 7 │      │
│[360, │[360, │[360, │     │[360, │[360, │[360, │      │
│1536] │1536] │1536] │     │1536] │1536] │1536] │      │
└──────┴──────┴──────┴─────┴──────┴──────┴──────┴──────┘
   ↑      ↑      ↑            ↑      ↑      ↑
  D0     D1     D2           D5     D6     D7
  D8     D9     D10         D13    D14    D15
  D16    D17    D18         D21    D22    D23
  D24    D25    D26         D29    D30    D31

Each column device in all rows holds same weight shard
```

### Computation with TP
```
Input on Each Device: [1, 1, N, 2880]
                           ↓
           Local Matmul with Weight Shard
                           ↓
Partial Output: [1, 1, N, 1536] (1/8 of full output)
                           ↓
             All-Reduce Across Columns
                           ↓
Complete Output: [1, 1, N, 12288] (full dimension)
```

## Expert MLP Processing Pipeline

### Per-Device Expert Processing
```
Device D0 (Row 0, Col 0):
┌────────────────────────────────┐
│ Experts: [0, 1, 2, 3]          │
│                                │
│ Expert 0: Process N₀ tokens    │
│ Expert 1: Process N₁ tokens    │
│ Expert 2: Process N₂ tokens    │
│ Expert 3: Process N₃ tokens    │
│                                │
│ Total: Σ(Nᵢ) tokens            │
└────────────────────────────────┘

Tensor Shape Evolution:
Input:  [1, 4, Σ(Nᵢ), 2880]
   ↓ Gate/Up Projection
Hidden: [1, 4, Σ(Nᵢ), 12288]
   ↓ SwiGLU
Active: [1, 4, Σ(Nᵢ), 6144]
   ↓ Down Projection
Output: [1, 4, Σ(Nᵢ), 2880]
```

## Communication Patterns Summary

### All-to-All (Expert Parallel)
```
Purpose: Token ↔ Expert routing
Direction: Along rows (axis=0)
Participants: 4 devices per all-to-all group
Pattern: Ring topology

     R0 ←→ R1
     ↑      ↓
     R3 ←→ R2
```

### All-Reduce (Tensor Parallel)
```
Purpose: Aggregate partial results
Direction: Along columns (axis=1)
Participants: 8 devices per all-reduce group
Pattern: Ring topology

C0 ← C1 ← C2 ← C3 ← C4 ← C5 ← C6 ← C7
↓                                    ↑
└────────────────────────────────────┘
```

## Dynamic Batching Visualization

### Traditional Static Expert Assignment
```
┌─────────────────────────────┐
│ Expert 0: Always 32 tokens  │ ← Fixed allocation
│ Expert 1: Always 32 tokens  │ ← May be underutilized
│ Expert 2: Always 32 tokens  │ ← Or oversubscribed
│ Expert 3: Always 32 tokens  │
└─────────────────────────────┘
Problem: Load imbalance, wasted compute
```

### Dynamic Batching with All-to-All
```
Time T1:                        Time T2:
┌─────────────────────┐        ┌─────────────────────┐
│ Expert 0: 45 tokens │        │ Expert 0: 12 tokens │
│ Expert 1: 18 tokens │        │ Expert 1: 67 tokens │
│ Expert 2: 52 tokens │        │ Expert 2: 23 tokens │
│ Expert 3: 13 tokens │        │ Expert 3: 26 tokens │
└─────────────────────┘        └─────────────────────┘
Benefit: Adapts to actual routing decisions
```

## Memory Hierarchy and Data Movement

### L1 Memory Layout (Decode)
```
Per Device L1 (1MB):
┌─────────────────────────────────┐
│ Input Activations: ~100KB       │
│ Weight Shards:     ~400KB       │
│ Output Buffer:     ~100KB       │
│ Workspace:         ~400KB       │
└─────────────────────────────────┘
Fast access, limited capacity
```

### DRAM Layout (Prefill)
```
Per Device DRAM (8GB available):
┌─────────────────────────────────┐
│ Input Activations: ~10MB        │
│ Weight Shards:     ~40MB        │
│ Output Buffer:     ~10MB        │
│ Workspace:         ~40MB        │
└─────────────────────────────────┘
Large capacity, higher latency
```

## Data Flow Timeline

```
Time →
T0: Input arrives
    ├─ Row sharding
T1: Router computation
    ├─ Top-K selection
T2: All-to-All dispatch START
    ├─ Ring step 1
    ├─ Ring step 2
    ├─ Ring step 3
T3: All-to-All dispatch COMPLETE
    ├─ Tokens at expert devices
T4: Expert MLP computation START
    ├─ Gate/Up projection
    ├─ SwiGLU activation
    ├─ Down projection
T5: Expert MLP computation COMPLETE
T6: All-to-All combine START
    ├─ Ring step 1
    ├─ Ring step 2
    ├─ Ring step 3
T7: All-to-All combine COMPLETE
    ├─ Outputs at original positions
T8: Weighted aggregation
    ├─ Apply routing weights
    ├─ Sum across experts
T9: All-Reduce START (TP)
    ├─ Reduce-scatter
    ├─ All-gather
T10: All-Reduce COMPLETE
     └─ Final output ready
```

## Parallelism Efficiency Analysis

### Expert Parallelism (EP=4)
```
Efficiency Factors:
┌──────────────────────────────────┐
│ + Distributes memory: 128/4 = 32 │
│   experts per row                │
│ + Dynamic load balancing via     │
│   all-to-all                     │
│ - Communication overhead: 3 ring │
│   steps for all-to-all          │
│ - Potential load imbalance if    │
│   routing is skewed             │
└──────────────────────────────────┘
```

### Tensor Parallelism (TP=8)
```
Efficiency Factors:
┌──────────────────────────────────┐
│ + Reduces memory per device by 8x│
│ + Enables larger models          │
│ + Efficient ring all-reduce      │
│ - Communication for every layer  │
│ - Latency increases with TP      │
└──────────────────────────────────┘
```

## Optimization Strategies

### Communication Optimization
```
1. Overlap Computation and Communication:
   While All-to-All for Layer N:
   └─ Compute non-MoE ops for Layer N-1

2. Ring Topology Benefits:
   - Balanced bandwidth usage
   - No congestion at single point
   - Scales linearly with devices

3. Batched Operations:
   - Group multiple small tensors
   - Reduce communication overhead
```

### Memory Optimization
```
1. L1 for Decode (Latency-Critical):
   Input → L1 → Compute → L1 → Output

2. DRAM for Prefill (Throughput):
   Input → DRAM → Stream → Compute → DRAM

3. Weight Reuse:
   - Keep frequently used experts in L1
   - Stream less-used experts from DRAM
```

## Scaling Analysis

### Scaling Expert Count
```
Current: 128 experts on 32 devices
         4 experts/device

Scale to 1024 experts:
Option 1: Increase EP to 32
         └─ 32 experts/device
         └─ More memory pressure

Option 2: Increase devices to 256 (16×16)
         └─ 4 experts/device maintained
         └─ Higher communication cost
```

### Scaling Batch Size
```
Current: 128 tokens, 32 per row

Scale to 1024 tokens:
- 256 tokens per row
- Same parallelism strategy
- Higher L1 memory usage
- May need DRAM for large batches
```

## Performance Bottlenecks

### Communication Bottlenecks
```
All-to-All Dispatch/Combine:
├─ Latency: O(EP-1) = O(3) steps
├─ Bandwidth: Ring utilization
└─ Solution: Overlap with compute

All-Reduce (TP):
├─ Latency: O(log TP) steps
├─ Bandwidth: Full ring bandwidth
└─ Solution: Fusion, larger blocks
```

### Compute Bottlenecks
```
Expert MLP:
├─ Matmul throughput
├─ Memory bandwidth (weights)
└─ Solution: Optimize configs, fusion

Load Imbalance:
├─ Some experts overloaded
├─ Others underutilized
└─ Solution: Load balancing loss
```
