# DeepSeek V3 MoE Parallelism Diagrams and Visualizations

## Device Mesh Visualizations

### TG Device Mesh (4×8 = 32 devices)

```
                        Tensor Parallelism (TP) Axis = 1
        ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
        │ TP=0 │ TP=1 │ TP=2 │ TP=3 │ TP=4 │ TP=5 │ TP=6 │ TP=7 │
┌───────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ EP=0  │ D0,0 │ D0,1 │ D0,2 │ D0,3 │ D0,4 │ D0,5 │ D0,6 │ D0,7 │ ← Experts 0-63
├───────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ EP=1  │ D1,0 │ D1,1 │ D1,2 │ D1,3 │ D1,4 │ D1,5 │ D1,6 │ D1,7 │ ← Experts 64-127
├───────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ EP=2  │ D2,0 │ D2,1 │ D2,2 │ D2,3 │ D2,4 │ D2,5 │ D2,6 │ D2,7 │ ← Experts 128-191
├───────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ EP=3  │ D3,0 │ D3,1 │ D3,2 │ D3,3 │ D3,4 │ D3,5 │ D3,6 │ D3,7 │ ← Experts 192-255
└───────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
        ↑
Expert Parallelism (EP) Axis = 0

Each device D[ep,tp] contains:
- 8 experts (256 experts / 32 devices)
- 1/8 of each expert's weights (TP sharding)
```

### QUAD Device Mesh (16×8 = 128 devices)

```
                        Tensor Parallelism (TP) Axis = 1
        ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
        │ TP=0 │ TP=1 │ TP=2 │ TP=3 │ TP=4 │ TP=5 │ TP=6 │ TP=7 │
┌───────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ EP=0  │ D0,0 │ D0,1 │ D0,2 │ D0,3 │ D0,4 │ D0,5 │ D0,6 │ D0,7 │ ← Experts 0-15
├───────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ EP=1  │ D1,0 │ D1,1 │ D1,2 │ D1,3 │ D1,4 │ D1,5 │ D1,6 │ D1,7 │ ← Experts 16-31
├───────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ EP=2  │ D2,0 │ D2,1 │ D2,2 │ D2,3 │ D2,4 │ D2,5 │ D2,6 │ D2,7 │ ← Experts 32-47
├───────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│  ...  │  ... │  ... │  ... │  ... │  ... │  ... │  ... │  ... │
├───────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ EP=14 │D14,0 │D14,1 │D14,2 │D14,3 │D14,4 │D14,5 │D14,6 │D14,7 │ ← Experts 224-239
├───────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ EP=15 │D15,0 │D15,1 │D15,2 │D15,3 │D15,4 │D15,5 │D15,6 │D15,7 │ ← Experts 240-255
└───────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
        ↑
Expert Parallelism (EP) Axis = 0

Each device D[ep,tp] contains:
- 2 experts (256 experts / 128 devices)
- 1/8 of each expert's weights (TP sharding)
```

## Expert Distribution Patterns

### TG Expert Distribution (32 devices, 8 experts/device)

```
Device Row EP=0 (Experts 0-63):
┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
│   D0,0     │   D0,1     │   D0,2     │   D0,3     │   D0,4     │   D0,5     │   D0,6     │   D0,7     │
│ Exp 0-7    │ Exp 0-7    │ Exp 0-7    │ Exp 0-7    │ Exp 0-7    │ Exp 0-7    │ Exp 0-7    │ Exp 0-7    │
│ Exp 8-15   │ Exp 8-15   │ Exp 8-15   │ Exp 8-15   │ Exp 8-15   │ Exp 8-15   │ Exp 8-15   │ Exp 8-15   │
│ Exp 16-23  │ Exp 16-23  │ Exp 16-23  │ Exp 16-23  │ Exp 16-23  │ Exp 16-23  │ Exp 16-23  │ Exp 16-23  │
│ Exp 24-31  │ Exp 24-31  │ Exp 24-31  │ Exp 24-31  │ Exp 24-31  │ Exp 24-31  │ Exp 24-31  │ Exp 24-31  │
│ Exp 32-39  │ Exp 32-39  │ Exp 32-39  │ Exp 32-39  │ Exp 32-39  │ Exp 32-39  │ Exp 32-39  │ Exp 32-39  │
│ Exp 40-47  │ Exp 40-47  │ Exp 40-47  │ Exp 40-47  │ Exp 40-47  │ Exp 40-47  │ Exp 40-47  │ Exp 40-47  │
│ Exp 48-55  │ Exp 48-55  │ Exp 48-55  │ Exp 48-55  │ Exp 48-55  │ Exp 48-55  │ Exp 48-55  │ Exp 48-55  │
│ Exp 56-63  │ Exp 56-63  │ Exp 56-63  │ Exp 56-63  │ Exp 56-63  │ Exp 56-63  │ Exp 56-63  │ Exp 56-63  │
└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘

Note: All devices in a row have the SAME experts (replicated across TP dimension)
```

### QUAD Expert Distribution (128 devices, 2 experts/device)

```
Device Row EP=0 (Experts 0-15):
┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
│   D0,0     │   D0,1     │   D0,2     │   D0,3     │   D0,4     │   D0,5     │   D0,6     │   D0,7     │
│ Exp 0-1    │ Exp 0-1    │ Exp 0-1    │ Exp 0-1    │ Exp 0-1    │ Exp 0-1    │ Exp 0-1    │ Exp 0-1    │
│ Exp 2-3    │ Exp 2-3    │ Exp 2-3    │ Exp 2-3    │ Exp 2-3    │ Exp 2-3    │ Exp 2-3    │ Exp 2-3    │
│ Exp 4-5    │ Exp 4-5    │ Exp 4-5    │ Exp 4-5    │ Exp 4-5    │ Exp 4-5    │ Exp 4-5    │ Exp 4-5    │
│ Exp 6-7    │ Exp 6-7    │ Exp 6-7    │ Exp 6-7    │ Exp 6-7    │ Exp 6-7    │ Exp 6-7    │ Exp 6-7    │
│ Exp 8-9    │ Exp 8-9    │ Exp 8-9    │ Exp 8-9    │ Exp 8-9    │ Exp 8-9    │ Exp 8-9    │ Exp 8-9    │
│ Exp 10-11  │ Exp 10-11  │ Exp 10-11  │ Exp 10-11  │ Exp 10-11  │ Exp 10-11  │ Exp 10-11  │ Exp 10-11  │
│ Exp 12-13  │ Exp 12-13  │ Exp 12-13  │ Exp 12-13  │ Exp 12-13  │ Exp 12-13  │ Exp 12-13  │ Exp 12-13  │
│ Exp 14-15  │ Exp 14-15  │ Exp 14-15  │ Exp 14-15  │ Exp 14-15  │ Exp 14-15  │ Exp 14-15  │ Exp 14-15  │
└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘

More fine-grained distribution with only 2 experts per device
```

## Grouped Routing Visualization

### Expert Groups Structure (256 experts = 8 groups × 32 experts/group)

```
┌─────────────────────────────────── 256 Total Experts ────────────────────────────────────┐
│                                                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │
│  │   Group 0   │  │   Group 1   │  │   Group 2   │  │   Group 3   │                    │
│  │ Experts     │  │ Experts     │  │ Experts     │  │ Experts     │                    │
│  │   0-31      │  │   32-63     │  │   64-95     │  │   96-127    │                    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘                    │
│                                                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │
│  │   Group 4   │  │   Group 5   │  │   Group 6   │  │   Group 7   │                    │
│  │ Experts     │  │ Experts     │  │ Experts     │  │ Experts     │                    │
│  │  128-159    │  │  160-191    │  │  192-223    │  │  224-255    │                    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘                    │
│                                                                                           │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

### Two-Stage Selection Process

```
Stage 1: Within-Group Selection (K=2 per group)
┌─────────────┐
│   Group 0   │ ──→ Select Top-2 ──→ [Expert_i, Expert_j]
└─────────────┘
┌─────────────┐
│   Group 1   │ ──→ Select Top-2 ──→ [Expert_k, Expert_l]
└─────────────┘
      ...              ...                   ...
┌─────────────┐
│   Group 7   │ ──→ Select Top-2 ──→ [Expert_y, Expert_z]
└─────────────┘

Result: 8 groups × 2 experts = 16 candidate experts

Stage 2: Group Selection (K=4 groups)
┌──────────────────────────────────────┐
│    16 Candidate Experts              │
│    (2 from each of 8 groups)         │
├──────────────────────────────────────┤
│  Select Top-4 Groups based on        │
│  maximum scores within each group    │
├──────────────────────────────────────┤
│  Final: 4 groups × 2 experts =       │
│         8 experts selected           │
└──────────────────────────────────────┘
```

## Communication Flow Diagrams

### All-to-All Dispatch Pattern

```
Before All-to-All (Tokens on source devices):
┌─────────────────────────────────────────────────────────┐
│ EP=0 │ Token A→Expert 150 │ Token B→Expert 45  │ ...   │
│ EP=1 │ Token C→Expert 12  │ Token D→Expert 200 │ ...   │
│ EP=2 │ Token E→Expert 88  │ Token F→Expert 150 │ ...   │
│ EP=3 │ Token G→Expert 45  │ Token H→Expert 12  │ ...   │
└─────────────────────────────────────────────────────────┘
                            ↓
                    All-to-All on EP Axis
                            ↓
After All-to-All (Tokens routed to expert devices):
┌─────────────────────────────────────────────────────────┐
│ EP=0 │ Tokens for Experts 0-63:   C, H, ...            │
│ EP=1 │ Tokens for Experts 64-127: B, G, E, ...         │
│ EP=2 │ Tokens for Experts 128-191: A, F, ...           │
│ EP=3 │ Tokens for Experts 192-255: D, ...              │
└─────────────────────────────────────────────────────────┘
```

### All-to-All Combine Pattern

```
After Expert Processing:
┌─────────────────────────────────────────────────────────┐
│ EP=0 │ Processed: C', H', ...  (from Experts 0-63)     │
│ EP=1 │ Processed: B', G', E', ... (from Experts 64-127)│
│ EP=2 │ Processed: A', F', ... (from Experts 128-191)   │
│ EP=3 │ Processed: D', ... (from Experts 192-255)       │
└─────────────────────────────────────────────────────────┘
                            ↓
                    All-to-All on EP Axis
                            ↓
After Combine (Outputs back to original positions):
┌─────────────────────────────────────────────────────────┐
│ EP=0 │ Token A' │ Token B' │ ...                       │
│ EP=1 │ Token C' │ Token D' │ ...                       │
│ EP=2 │ Token E' │ Token F' │ ...                       │
│ EP=3 │ Token G' │ Token H' │ ...                       │
└─────────────────────────────────────────────────────────┘
```

### Tensor Parallel Communication

```
TP All-Gather (Before Router):
┌──────────┬──────────┬──────────┬────┬──────────┐
│  TP=0    │  TP=1    │  TP=2    │... │  TP=7    │
│ [B,S,896]│ [B,S,896]│ [B,S,896]│... │ [B,S,896]│
└──────────┴──────────┴──────────┴────┴──────────┘
                        ↓
                   All-Gather
                        ↓
┌─────────────────────────────────────────────────┐
│         All Devices: [B,S,7168]                 │
└─────────────────────────────────────────────────┘

TP Reduce-Scatter (After MoE):
┌─────────────────────────────────────────────────┐
│         All Devices: [B,S,7168]                 │
└─────────────────────────────────────────────────┘
                        ↓
                 Reduce-Scatter
                        ↓
┌──────────┬──────────┬──────────┬────┬──────────┐
│  TP=0    │  TP=1    │  TP=2    │... │  TP=7    │
│ [B,S,896]│ [B,S,896]│ [B,S,896]│... │ [B,S,896]│
└──────────┴──────────┴──────────┴────┴──────────┘
```

## Complete Tensor Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Input (TP-sharded)                          │
│                      [batch, 1, seq_len, 896]                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓ TP All-Gather
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                      Full Input Reconstructed                       │
│                     [batch, 1, seq_len, 7168]                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
         ┌──────────────────┐            ┌──────────────────┐
         │ GroupedTopKRouter│            │  Shared Expert   │
         │   (MoEGate)      │            │   Computation    │
         └──────────────────┘            └──────────────────┘
                    │                               │
                    ↓                               │
         ┌──────────────────┐                      │
         │   MoE Preamble   │                      │
         │  (Preprocessing) │                      │
         └──────────────────┘                      │
                    │                               │
                    ↓                               │
         ┌──────────────────┐                      │
         │  All-to-All      │                      │
         │   Dispatch       │                      │
         └──────────────────┘                      │
                    │                               │
                    ↓                               │
         ┌──────────────────┐                      │
         │ Routed Experts   │                      │
         │  (256 MLPs)      │                      │
         └──────────────────┘                      │
                    │                               │
                    ↓                               │
         ┌──────────────────┐                      │
         │  All-to-All      │                      │
         │   Combine        │                      │
         └──────────────────┘                      │
                    │                               │
                    ↓                               ↓
                    └───────────────┬───────────────┘
                                    │
                                    ↓ Combine + Apply Weights
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                         Full MoE Output                             │
│                     [batch, 1, seq_len, 7168]                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓ TP Reduce-Scatter
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                        Output (TP-sharded)                          │
│                      [batch, 1, seq_len, 896]                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Scaling Analysis

### Expert Distribution Scaling

```
┌────────────┬──────────┬──────────┬──────────────┬─────────────────┐
│   Config   │ Devices  │   EP×TP  │ Experts/Dev  │ Memory/Device   │
├────────────┼──────────┼──────────┼──────────────┼─────────────────┤
│    TG      │    32    │   4×8    │      8       │    ~480MB       │
│   QUAD     │   128    │  16×8    │      2       │    ~120MB       │
│  Future    │   256    │  32×8    │      1       │     ~60MB       │
└────────────┴──────────┴──────────┴──────────────┴─────────────────┘

Scaling Observations:
- Linear reduction in experts per device
- Memory per device decreases with scale
- Communication overhead increases with EP
```

### Communication Volume Analysis

```
All-to-All Communication (per token):
┌────────────┬────────────────┬─────────────────┬──────────────────┐
│   Config   │  Dispatch Vol  │  Combine Vol    │     Total        │
├────────────┼────────────────┼─────────────────┼──────────────────┤
│    TG      │  8×7168 floats │  8×7168 floats  │  ~450KB/token    │
│   QUAD     │  8×7168 floats │  8×7168 floats  │  ~450KB/token    │
└────────────┴────────────────┴─────────────────┴──────────────────┘

TP Communication (per sequence):
┌────────────┬────────────────┬─────────────────┬──────────────────┐
│   Config   │   All-Gather   │ Reduce-Scatter  │     Total        │
├────────────┼────────────────┼─────────────────┼──────────────────┤
│  TG/QUAD   │  S×7168 floats │  S×7168 floats  │  ~56KB×S/token   │
└────────────┴────────────────┴─────────────────┴──────────────────┘

Where S = sequence length
```

### Latency Breakdown

```
Decode Mode (1 token) Latency Components:
┌─────────────────────────────────────────┐
│ Component           │ Estimated Time    │
├─────────────────────┼───────────────────┤
│ TP All-Gather       │     ~0.5ms       │
│ Router Computation  │     ~0.2ms       │
│ MoE Preamble        │     ~0.1ms       │
│ All-to-All Dispatch │     ~1.0ms       │
│ Expert Computation  │     ~2.0ms       │
│ All-to-All Combine  │     ~1.0ms       │
│ TP Reduce-Scatter   │     ~0.5ms       │
├─────────────────────┼───────────────────┤
│ Total               │     ~5.3ms       │
└─────────────────────┴───────────────────┘

Prefill Mode (128 tokens) Latency:
- Approximately 128× decode latency
- Memory bandwidth becomes limiting factor
- Batching improves throughput efficiency
```

## Memory Layout Visualization

### Expert Weight Storage (Per Device)

```
TG Device (8 experts):
┌────────────────────────────────────────────┐
│             Device Memory (DRAM)            │
├────────────────────────────────────────────┤
│  Expert 0 Weights:                         │
│    W_gate:  [7168/8, 2048/8] = [896, 256]  │
│    W_up:    [7168/8, 2048/8] = [896, 256]  │
│    W_down:  [2048/8, 7168/8] = [256, 896]  │
├────────────────────────────────────────────┤
│  Expert 1 Weights: (same dimensions)       │
├────────────────────────────────────────────┤
│  ...                                        │
├────────────────────────────────────────────┤
│  Expert 7 Weights: (same dimensions)       │
├────────────────────────────────────────────┤
│  Shared Expert Weights (1/8 portion):      │
│    W_gate:  [7168/8, 10752/8]              │
│    W_up:    [7168/8, 10752/8]              │
│    W_down:  [10752/8, 7168/8]              │
└────────────────────────────────────────────┘

QUAD Device (2 experts):
┌────────────────────────────────────────────┐
│             Device Memory (DRAM)            │
├────────────────────────────────────────────┤
│  Expert 0 Weights:                         │
│    W_gate, W_up, W_down (TP-sharded)       │
├────────────────────────────────────────────┤
│  Expert 1 Weights:                         │
│    W_gate, W_up, W_down (TP-sharded)       │
├────────────────────────────────────────────┤
│  Shared Expert Weights (1/8 portion)       │
├────────────────────────────────────────────┤
│  More space for activations                │
└────────────────────────────────────────────┘
```

## Optimization Opportunities

### Current Bottlenecks and Potential Optimizations

```
┌──────────────────────┬────────────────────┬─────────────────────────┐
│    Bottleneck        │  Current Impact    │  Optimization           │
├──────────────────────┼────────────────────┼─────────────────────────┤
│ All-to-All Latency   │   ~40% of time     │ Overlap with compute    │
│ Router Computation   │   Sequential       │ Parallelize scoring     │
│ Memory Bandwidth     │   Prefill limited  │ Activation compression  │
│ Load Imbalance       │   Variable         │ Auxiliary loss tuning   │
└──────────────────────┴────────────────────┴─────────────────────────┘
```

### Future Scaling Considerations

```
256-Device Configuration (32×8):
┌────────────────────────────────────────────────┐
│ • 1 expert per device                         │
│ • Minimal memory footprint                    │
│ • Maximum communication overhead              │
│ • Ideal for very large models                 │
└────────────────────────────────────────────────┘

512-Device Configuration (64×8):
┌────────────────────────────────────────────────┐
│ • Multiple devices per expert                 │
│ • Expert replication for redundancy           │
│ • Advanced routing strategies needed          │
│ • Hierarchical all-to-all patterns            │
└────────────────────────────────────────────────┘
```

## Summary

The DeepSeek V3 MoE parallelism strategy effectively distributes 256 experts across device meshes using a combination of expert parallelism (EP) and tensor parallelism (TP). The hierarchical GroupedTopK routing mechanism reduces selection complexity while maintaining model quality. The architecture scales efficiently from 32 devices (TG) to 128 devices (QUAD) and beyond, with careful orchestration of all-to-all communication patterns and memory optimization strategies.

Key insights:
1. **Grouped routing** reduces computational overhead through two-stage selection
2. **Shared expert** maintains general knowledge while routed experts specialize
3. **EP/TP split** balances communication and memory requirements
4. **Scaling** from TG to QUAD shows near-linear efficiency for large batches
5. **Communication patterns** are well-optimized for the hardware topology
