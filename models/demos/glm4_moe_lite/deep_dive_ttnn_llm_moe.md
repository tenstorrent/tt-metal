# Deep Dive: LLM Implementations on TTNN — Parallelization, CCLs, and Expert-Parallel MoE

**Date:** March 17, 2026
**Sources:** `tech_reports/LLMs/llms.md`, `understanding_ttnn_ccls_for_ep_moe.md`, `tech_reports/LLMs/vLLM_integration.md`
**Target hardware:** Wormhole-based systems — N300 (2 chips), T3K/T3000 (8 chips), TG/Galaxy (32 chips)

---

## Table of Contents

1. [Part 1: Hardware Foundation and Memory Hierarchy](#part-1-hardware-foundation-and-memory-hierarchy)
2. [Part 2: Multi-Device Weight Parallelization Strategies](#part-2-multi-device-weight-parallelization-strategies)
3. [Part 3: Collective Communication Library (CCL) Operations](#part-3-collective-communication-library-ccl-operations)
4. [Part 4: Dense MLP vs Mixture-of-Experts on TTNN](#part-4-dense-mlp-vs-mixture-of-experts-on-ttnn)
5. [Part 5: Expert-Parallel MoE — The Full Picture](#part-5-expert-parallel-moe--the-full-picture)
6. [Part 6: Serving Integration — Continuous Batching, vLLM, and Tracing](#part-6-serving-integration--continuous-batching-vllm-and-tracing)
7. [Part 7: Performance Optimization Guide](#part-7-performance-optimization-guide)
8. [Appendices](#appendices)

---

## Part 1: Hardware Foundation and Memory Hierarchy

Understanding how data lives and moves on Tenstorrent hardware is prerequisite to every parallelization decision that follows.

### 1.1 The MeshDevice — A 2D Grid of Independent Chips

Each Tenstorrent multi-chip system is a **MeshDevice**: a 2D grid of Wormhole chips connected by Ethernet. Each chip has its own DRAM, L1 SRAM, and 64 Tensix compute cores. There is **no shared memory** — data moves between chips exclusively over Ethernet links.

```
Board Configurations:

  N150:   1 chip    mesh (1,1)
  N300:   2 chips   mesh (1,2)     single board, 2 Wormhole chips
  T3K:    8 chips   mesh (1,8)     4 x N300 boards
  Galaxy: 32 chips  mesh (4,8)     4 rows x 8 columns
```

```
T3K mesh (1, 8):

              axis 1  (columns, 8 devices)
   +-------------------------------------------------------------+
   |  +----+  ETH  +----+  ETH  +----+        +----+  ETH  +----+|
   |  | D0 |<----->| D1 |<----->| D2 |  ...  | D6 |<----->| D7 ||
   |  |DRAM|       |DRAM|       |DRAM|        |DRAM|       |DRAM||
   |  | L1 |       | L1 |       | L1 |        | L1 |       | L1 ||
   |  |64  |       |64  |       |64  |        |64  |       |64  ||
   |  |cors|       |cors|       |cors|        |cors|       |cors||
   |  +----+       +----+       +----+        +----+       +----+|
   +-------------------------------------------------------------+

Galaxy/TG mesh (4, 8):

       col 0   col 1   col 2   col 3   col 4   col 5   col 6   col 7
      +-----+ +-----+ +-----+ +-----+ +-----+ +-----+ +-----+ +-----+
row 0 | D0  |-| D1  |-| D2  |-| D3  |-| D4  |-| D5  |-| D6  |-| D7  |
      +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+
      +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+
row 1 | D8  |-| D9  |-| D10 |-| D11 |-| D12 |-| D13 |-| D14 |-| D15 |
      +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+
      +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+
row 2 | D16 |-| D17 |-| D18 |-| D19 |-| D20 |-| D21 |-| D22 |-| D23 |
      +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+
      +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+ +--+--+
row 3 | D24 |-| D25 |-| D26 |-| D27 |-| D28 |-| D29 |-| D30 |-| D31 |
      +-----+ +-----+ +-----+ +-----+ +-----+ +-----+ +-----+ +-----+
```

### 1.2 Per-Chip Memory Hierarchy

Within a single Wormhole chip, there is a two-level memory hierarchy that dominates every design decision:

```
+------------------------- Single Wormhole Chip -------------------------+
|                                                                        |
|   +----------------------------+                                       |
|   |         DRAM               |  ~12 GB, ~240 GB/s (sharded),        |
|   |   12 banks, off-chip       |  ~190 GB/s (interleaved)             |
|   +------------+---------------+  Stores: weights, KV cache, prefill  |
|                | NOC                                                   |
|   +------------+---------------+                                       |
|   |     64 Tensix Cores        |                                       |
|   |  +------+ +------+  ...   |  ~1.5 MB L1 per core, ultra fast     |
|   |  | L1   | | L1   |       |  Stores: decode activations,          |
|   |  | SRAM | | SRAM |       |  intermediate results                  |
|   |  |+Math | |+Math |       |  L1 local read = zero NOC cost        |
|   |  +------+ +------+       |                                       |
|   +----------------------------+                                       |
+------------------------------------------------------------------------+
```

**The fundamental bottleneck differs by mode:**

| Mode | Bottleneck | Why |
|------|-----------|-----|
| **Decode** (seq_len=1) | **DRAM bandwidth** — reading weights | Activations are tiny [batch, hidden_dim]; compute is fast. The limiter is streaming weights from DRAM. |
| **Prefill** (seq_len=N) | **Compute** — matmul throughput | Both activations and weights are large; math engines are the limiter. |

This is why decode and prefill use fundamentally different strategies:

```
+------------------------------+--------------------------------------+
|         DECODE               |           PREFILL                    |
+------------------------------+--------------------------------------+
| Activations: L1 sharded     | Activations: DRAM interleaved        |
| Weights: DRAM sharded       | Weights: DRAM interleaved            |
| Matmul: DRAM-sharded        | Matmul: 2D multi-core                |
| Bottleneck: DRAM BW         | Bottleneck: Compute                  |
| KV cache: DRAM paged        | KV cache: DRAM paged (fill)          |
| Tracing: YES                | Tracing: NO (variable seq_len)       |
+------------------------------+--------------------------------------+
```

### 1.3 Data Placement: Four Levels of Speed

```
Speed Ranking (slowest to fastest):

  1. DRAM Interleaved    ~190 GB/s   Each tile round-robins across DRAM banks
  2. DRAM Sharded        ~240 GB/s   Tiles collocated with nearest compute core
  3. L1 Interleaved      faster      Tiles spread across all cores L1 via NOC
  4. L1 Sharded          fastest     Tiles on the EXACT core that needs them
```

The golden rule: **minimize data movement**. The ideal OP reads L1-sharded input from the core it is already on, computes, and writes L1-sharded output that the next OP can consume without any reshard.

---

## Part 2: Multi-Device Weight Parallelization Strategies

When a model is too large for a single chip, weights must be distributed across devices. The choice of parallelization strategy determines what CCL operations are needed and the total data movement volume.

### 2.1 Tensor Distribution: Replicate vs Shard

Before computation, host tensors must be placed onto the mesh:

```
Strategy 1: ReplicateTensorToMesh — Full copy on every device

  Host tensor [1,1,T,H]
       |
       +---> D0: [1,1,T,H]  (full copy)
       +---> D1: [1,1,T,H]  (full copy)
       +---> D2: [1,1,T,H]  (full copy)
       |       ...
       +---> D7: [1,1,T,H]  (full copy)

  Cost: 8x memory redundancy
  Use:  Activations every device needs (tensor-parallel decode)

Strategy 2: ShardTensorToMesh — Each device gets a slice

  Host tensor [1,1,T, 8*K]
       |
       +---> D0: [1,1,T,K]  (columns 0..K)
       +---> D1: [1,1,T,K]  (columns K..2K)
       |       ...
       +---> D7: [1,1,T,K]  (columns 7K..8K)

  Cost: 1x total memory (no redundancy)
  Use:  Expert weights in MoE, weight matrices in tensor-parallel
```

### 2.2 Sharding Schemes for Decode-Mode Weight Distribution

In decode mode, activations are small ([batch, hidden_dim], batch <= 32, seq_len = 1) and stored in L1. Weights are large and stored in DRAM. Sharding weights across devices reduces per-device DRAM pressure.

#### 2.2.1 1D Column Parallel

Weights are sharded in **width** (output dimension N). Each device holds a horizontal slice. Activations must be gathered (replicated) so each device processes the full activation.

```
                        Activation (gathered/replicated)
                        [batch, K]   full width on each device
                            |
        +-------------------+-------------------+
        v                   v                   v
   +---------+        +---------+        +---------+
   | W_shard0|        | W_shard1|        | W_shard2|     ...
   | [K,N/D] |        | [K,N/D] |        | [K,N/D] |
   +----+----+        +----+----+        +----+----+
        v                   v                   v
   Out_shard0          Out_shard1          Out_shard2
   [batch,N/D]         [batch,N/D]         [batch,N/D]

   Output: Width-fractured across devices
   CCL to restore: AllGather(dim=width)
```

#### 2.2.2 1D Row Parallel

Weights are sharded in **height** (reduction dimension K). Each device holds a vertical slice. Activations must be width-fractured.

```
   Act_shard0         Act_shard1         Act_shard2
   [batch,K/D]        [batch,K/D]        [batch,K/D]
        |                   |                   |
        v                   v                   v
   +---------+        +---------+        +---------+
   | W_shard0|        | W_shard1|        | W_shard2|
   | [K/D,N] |        | [K/D,N] |        | [K/D,N] |
   +----+----+        +----+----+        +----+----+
        v                   v                   v
   Partial_0           Partial_1           Partial_2
   [batch, N]          [batch, N]          [batch, N]

   Each output is a PARTIAL. Must be summed across devices.
   CCL: ReduceScatter (sum + reshard) or AllReduce (sum + replicate)
```

#### 2.2.3 1D Column + Row Parallel (1D Weight Sharding)

The most important optimization for MLP architectures. Combines column-parallel and row-parallel to **defer** the CCL to a more favorable point.

```
  The MLP Case (Llama3-70b: FF1/FF3 = [8K,28K], FF2 = [28K,8K]):

  Input x: [batch, 8K]  (gathered)
       |
       v
  FF1 Column-Parallel:  [batch, 8K] x [8K, 28K/D]
       |
       v
  Output: [batch, 28K/D]  (width-fractured)
       |
       v    <-- NO CCL HERE (fractured output feeds FF2 directly)
  FF2 Row-Parallel:     [batch, 28K/D] x [28K/D, 8K]
       |
       v
  Output: [batch, 8K]  PARTIAL
       |
       v    <-- CCL HERE: AllReduce or ReduceScatter+AllGather
  Final: [batch, 8K]  (gathered)

  Savings: CCL operates on [batch, 8K] instead of [batch, 28K] = 3.5x less data!
```

#### 2.2.4 2D Weight Sharding (Galaxy/TG)

On a 2D mesh like (4, 8), weights are block-sharded in both width and height:

```
  Galaxy mesh (4,8): axis=0 has 4 devices, axis=1 has 8 devices

  Weight [K, N] sharded as [K/4, N/8] per device:

       +--------+--------+--- ... ---+--------+
       |(0,0)   |(0,1)   |           |(0,7)   |   8 column shards of N
  K/4  |K/4,N/8 |K/4,N/8 |           |K/4,N/8 |
       +--------+--------+--- ... ---+--------+
       |(1,0)   |(1,1)   |           |(1,7)   |
  K/4  |        |        |           |        |
       +--------+--------+--- ... ---+--------+
       |(2,0)   |(2,1)   |           |(2,7)   |
  K/4  |        |        |           |        |
       +--------+--------+--- ... ---+--------+
       |(3,0)   |(3,1)   |           |(3,7)   |
  K/4  |        |        |           |        |
       +--------+--------+--- ... ---+--------+

  Activations: width-fractured along axis=0, replicated along axis=1
  Output: width-fractured along axis=0, PARTIAL along axis=1
  CCL: AllReduce along axis=1 to accumulate partials
```

### 2.3 Choosing the Optimal Strategy

The optimal parallelization strategy minimizes total CCL data movement:

| CCL Operation | Data Movement (Line) | Data Movement (Ring) |
|--------------|-------------------------------|-------------------------------|
| AllGather | (K*N*DF/D)*(D-1)*D | (K*N*DF)*D*log2(D) |
| ReduceScatter | (K*N*DF)*(1-1/D) | (K*N*DF)*(D-1)/D |

Where K,N = weight dimensions, DF = bytes per element, D = devices along CCL axis.

| Strategy | Input Requirement | Output Requirement |
|----------|------------------|-------------------|
| 1D Column Parallel | Gathered in K | Fractured in K |
| 1D Row Parallel | Fractured in K | Partials of full size |
| 1D Column + Row | Gathered in K | Partials of full size |
| 2D Parallel | Fractured in K | Partials over one axis |

### 2.4 Concrete Example: Llama3 Parallelization

| Matmul | N300 | T3000 | TG (Galaxy) |
|--------|------|-------|-------------|
| QKV projection | Column Parallel | Column Parallel | 2D |
| Dense out | Row Parallel | Row Parallel | 2D |
| FF1 (gate) | Column Parallel | Column Parallel | 2D |
| FF3 (up) | Column Parallel | Column Parallel | 2D |
| FF2 (down) | Row Parallel | Row Parallel | 2D |

---

## Part 3: Collective Communication Library (CCL) Operations

Since devices have no shared memory, every cross-device data exchange requires an explicit CCL operation over Ethernet.

### 3.1 The Three Universal Parameters

Every CCL op accepts three key parameters:

| Parameter | Description |
|-----------|------------|
| `cluster_axis` | Which mesh dimension to communicate along. axis=1 on (1,8): all 8 communicate. axis=0 on (4,8): 8 groups of 4 each. |
| `num_links` | Physical Ethernet connections per hop. 1 = safe default. TG: 4 on axis=0, 3 on axis=1. |
| `topology` | Linear (chain, safe default) or Ring (loop, better BW, needs wrap-around link). |

### 3.2 AllGather

Collects data from all devices and concatenates along a dimension. Every device ends up with the full tensor.

```
BEFORE AllGather (dim=3, 4 devices):
  D0: [batch, N/4]
  D1: [batch, N/4]      ==> AllGather(dim=3)
  D2: [batch, N/4]
  D3: [batch, N/4]

AFTER:
  ALL devices: [batch, N]   (full concatenation)
```

### 3.3 ReduceScatter

Reduces (sums) across devices and distributes different shards of the result.

```
BEFORE ReduceScatter (dim=3, 4 devices):
  D0: [batch, N]  partial_0
  D1: [batch, N]  partial_1      ==> ReduceScatter(dim=3, op=Sum)
  D2: [batch, N]  partial_2
  D3: [batch, N]  partial_3

AFTER:
  D0: [batch, N/4]   shard 0 of (partial_0+1+2+3)
  D1: [batch, N/4]   shard 1 of the sum
  D2: [batch, N/4]   shard 2 of the sum
  D3: [batch, N/4]   shard 3 of the sum
```

### 3.4 AllReduce

Reduces across devices and replicates the full result. Logically = ReduceScatter + AllGather.

```
BEFORE AllReduce (4 devices):
  D0: partial_0
  D1: partial_1      ==> AllReduce(op=Sum)
  D2: partial_2
  D3: partial_3

AFTER:
  ALL devices: partial_0 + partial_1 + partial_2 + partial_3  (full, replicated)
```

Physical implementation (Linear topology):
```
Step 1 (forward accumulation):
  D0 --[p0]--> D1 adds p1: [p0+p1] --> D2 adds: [p0+p1+p2] --> ... --> D7 has TOTAL

Step 2 (backward broadcast):
  D7 --[TOTAL]--> D6 --> D5 --> ... --> D0
  Now ALL devices have the total.
```

### 3.5 Relationship Between Standard CCLs

```
  AllReduce  =  ReduceScatter  +  AllGather

  ReduceScatter: each device gets a DIFFERENT shard of sum
  AllGather:     each device gets the FULL concatenation
  AllReduce:     each device gets the FULL sum

  Use ReduceScatter when downstream can consume shards
  Use AllReduce when downstream needs the complete tensor
  Use AllGather when collecting shards (no reduction needed)
```

### 3.6 MoE-Specific CCLs

These are NOT generic math collectives. They understand expert routing and physically move tokens to/from the devices that own the selected experts.

**all_to_all_dispatch:**
```
BEFORE:
  D0 has tokens [t0,t1]:  t0->expert 12 (D1), t1->expert 3 (D0)
  D1 has tokens [t2,t3]:  t2->expert 0 (D0), t3->expert 9 (D1)

             ==> all_to_all_dispatch (guided by expert_mapping_tensor)

AFTER:
  D0 receives: [t1, t2]  both need experts 0-7
  D1 receives: [t0, t3]  both need experts 8-15
  Also returns: dispatch_metadata (for reverse trip)
```

**all_to_all_combine:**
```
AFTER expert compute:
  D0 has results for [t1, t2]
  D1 has results for [t0, t3]

             ==> all_to_all_combine (uses dispatch_metadata)

AFTER:
  D0: results for [t0, t1]   (its original tokens)
  D1: results for [t2, t3]   (its original tokens)
```

### 3.7 CCL Operation Summary Table

| Operation | What happens | Output size vs input |
|-----------|-------------|---------------------|
| AllGather | Shards -> Full concat on all devices | Larger |
| ReduceScatter | Full tensors -> Sum -> Different shard per device | Smaller |
| AllReduce | Full tensors -> Sum -> Full result on all devices | Same |
| all_to_all_dispatch | MoE: tokens -> expert-home devices | Variable |
| all_to_all_combine | MoE: expert results -> originating devices | Variable |

### 3.8 CCL Usage in GLM4-MoE-Lite: Concrete Mapping

The GLM4-MoE-Lite implementation (`models/demos/glm4_moe_lite/`) provides a concrete case study of how CCL ops map to different parts of a real MoE model. Four of the five CCL ops appear in the codebase:

| CCL Op | File(s) | Call Sites | Purpose |
|--------|---------|------------|---------|
| `ttnn.all_reduce` | `moe_tt.py`, `attention_decode.py`, `mlp_decode.py`, `decoder_layer_tt.py`, `linear_helpers.py` | ~12 | Sum partials from TP or EP computation |
| `ttnn.all_gather` | `model_tt.py` | 2 | Concatenate sharded vocabulary logits in LM Head |
| `ttnn.all_to_all_dispatch` | `moe_tt.py` | 1 | Route tokens to expert-home devices (a2a path only) |
| `ttnn.all_to_all_combine` | `moe_tt.py` | 1 | Return expert results to originating devices (a2a path only) |
| `ttnn.reduce_scatter` | — | 0 | **Not used** in this model |

**Where each op appears and why:**

**`all_reduce` (the workhorse):**
- **Attention** (`attention_decode.py`) — After the output projection `w_o`, which is row-parallel, produces partials on each device. `all_reduce` combines them into the full attention output.
- **MLP / Shared Expert** (`mlp_decode.py`, `decoder_layer_tt.py`) — After each row-parallel FF2 matmul (for the routed MLP expert and the shared expert), partials are reduced across devices.
- **MoE reduce path** (`moe_tt.py`) — The final step of expert-parallel MoE: each device has computed its local experts' contributions via `sparse_matmul`, and `all_reduce` sums the 8 partial results so every device has the complete MoE output.
- **MoE a2a path** (`moe_tt.py`) — Even the a2a path uses `all_reduce` at the end on the secondary cluster axis (for 2D meshes) or when finishing the combine step.
- **Generic TP linear** (`linear_helpers.py`) — Reusable helper that wraps a column-parallel matmul followed by `all_reduce`.

**`all_gather` (LM Head only):**
- **LM Head** (`model_tt.py`) — The vocabulary weight matrix is sharded across devices. After each device computes its logit shard, `all_gather(dim=3)` concatenates them into the full vocabulary-width logits. This is the only place `all_gather` is used.

**`all_to_all_dispatch` + `all_to_all_combine` (MoE a2a path only):**
- **MoE a2a dispatch** (`moe_tt.py` line 1385) — Sends tokens to devices owning their selected experts based on the expert mapping tensor.
- **MoE a2a combine** (`moe_tt.py` line 1662) — Reverses the routing, sending expert computation results back to the devices that originally owned each token. Uses the opaque `dispatch_metadata` returned by `all_to_all_dispatch`.
- These are only invoked when `dispatch_impl="a2a"`, which is NOT the default production path.

**Why `reduce_scatter` is absent:**

Unlike Llama3's dense MLP (which uses `reduce_scatter` to produce width-fractured output after FF2), GLM4-MoE-Lite uses `all_reduce` everywhere. This keeps the output **replicated** across devices after every reduction, which is required because:
1. The EP reduce path needs replicated tokens for the next layer's router to produce identical results on all devices.
2. The attention layer similarly expects replicated input.
3. The trade-off is higher CCL data volume (full tensor replicated vs sharded), but it eliminates the need for a subsequent `all_gather` before the next consumer, simplifying the pipeline.

```
GLM4-MoE-Lite CCL flow per decoder layer (reduce path, T3K):

  Input (replicated) ---> Attention
                              |
                         w_qkv (column-par, no CCL)
                         RoPE, KV cache, SDPA
                         w_o (row-par) --> partial
                              |
                         all_reduce  <--- CCL #1
                              |
                         Residual add
                              |
                     --> MoE Layer
                              |
                         Router (replicated, no CCL)
                         scatter + remap (per-device)
                         sparse_matmul x 3 (per-device)
                         local aggregation
                              |
                         all_reduce  <--- CCL #2
                              |
                     --> Shared Expert (if present)
                              |
                         FF1+FF3 (column-par, no CCL)
                         SiLU * multiply
                         FF2 (row-par) --> partial
                              |
                         all_reduce  <--- CCL #3
                              |
                         Residual add --> Output (replicated)

  Total CCL ops per layer: 3 all_reduce calls
  (Plus 2 all_gather calls total in LM Head at the very end)
```

### 3.9 Why Both "reduce" and "a2a" Paths Exist

A common question: if the a2a path is not the default and performs worse for the current deployment, why is it in the codebase at all?

The answer is that each path is the **only correct or efficient** solution under different token distribution scenarios. Both exist because the model must support both deployment modes.

#### 3.9.1 Today: vLLM + T3K Decode (Replicated Tokens)

vLLM sends tokens to the model **replicated** — every device has the same `[batch, 1, hidden]`. Under this condition, `all_to_all_dispatch` is counterproductive:

```
What happens if you use a2a with replicated tokens (8 devices, batch=32):

  D0 has tokens [t0..t31]    (all 32 users)
  D1 has tokens [t0..t31]    (same 32 users, COPIES)
  ...
  D7 has tokens [t0..t31]    (same 32 users, COPIES)

  all_to_all_dispatch: "send each token to the device owning its expert"

  But EVERY device already has every token. So:
    D0 receives tokens from D0,D1,...,D7 = 8 copies of the same tokens
    Token count inflates: 32 -> 256 effective tokens per device

  Then sparse_matmul processes 256 tokens instead of 32 = 8x more work
  Then all_to_all_combine sends 256 results back = another Ethernet round trip

  Total: 2 CCL ops + 8x compute inflation = TERRIBLE for decode
```

The reduce path avoids all of this:
```
  Each device already has all tokens (replicated).
  Each device runs sparse_matmul on just its 8 local experts (32 tokens).
  One all_reduce sums the partials. Done.

  Total: 1 CCL op, no token inflation
```

This is why the code defaults to `dispatch_impl="reduce"` and the docstring explicitly warns about token inflation:

```python
# moe_tt.py lines 1191-1199 (docstring)
# - all_to_all_dispatch expects input tokens to be sharded across the
#   dispatch axis. In our vLLM bring-up (replicated activations on a mesh),
#   using all-to-all can inflate the effective token count and crater
#   decode throughput.
# - Default path is therefore a replicated-token strategy.
# - The older all-to-all dispatch/combine path remains available behind
#   GLM4_MOE_LITE_MOE_SPARSE_DISPATCH_IMPL=a2a for future DP-sharded inputs.
```

#### 3.9.2 Future: Data-Parallel Serving (Sharded Tokens)

Imagine a future deployment with 256 concurrent users, where the batch is **sharded** across devices:

```
  D0 has tokens for users 0-31      (32 UNIQUE users)
  D1 has tokens for users 32-63     (32 DIFFERENT users)
  D2 has tokens for users 64-95     (32 DIFFERENT users)
  ...
  D7 has tokens for users 224-255   (32 DIFFERENT users)
```

Now the reduce path **cannot work**:
```
  D0 only has users 0-31.
  D0's local experts are 0-7.
  But user 5 on D0 needs expert 40 (lives on D5).
  D0 does NOT have expert 40's weights. It cannot compute the answer.
  D5 does NOT have user 5's hidden states. It cannot compute either.

  Someone has to move either the tokens or the weights across Ethernet.
  Moving tokens (a2a dispatch) is far cheaper than moving weights.
```

With sharded tokens, the a2a path is the only viable solution:
```
  all_to_all_dispatch:
    D0 sends user 5's hidden states to D5 (where expert 40 lives)
    D5 sends user 200's hidden states to D0 (where expert 3 lives)
    No inflation: each token moves to exactly 1 destination device

  sparse_matmul: each device processes only the tokens actually routed to it

  all_to_all_combine: results go back to originating devices

  Total: 2 CCL ops, but ZERO redundancy in compute or memory
```

#### 3.9.3 Decision Table: When Each Path Wins

| Deployment Scenario | Token Distribution | Better Path | Why |
|--------------------|--------------------|-------------|-----|
| vLLM decode (today) | Replicated (all devices have all tokens) | **reduce** | 1 CCL op, no inflation |
| Data-parallel serving (future) | Sharded (each device has unique tokens) | **a2a** | Only way to route tokens to expert-home devices |
| Large-batch prefill (future) | Sharded | **a2a** | Memory savings (no 8x replication of long sequences) |
| Single device | N/A | Neither | No CCL needed, direct compute |

The environment variable `GLM4_MOE_LITE_MOE_SPARSE_DISPATCH_IMPL` is the switch:
- Unset or `"reduce"` (default): replicated-token path with `all_reduce`
- `"a2a"` or `"all_to_all"`: sharded-token path with `all_to_all_dispatch` + `all_to_all_combine`

When the serving stack evolves to support data-parallel sharded batches, flipping that flag activates the a2a path without any model code changes.

---

## Part 4: Dense MLP vs Mixture-of-Experts on TTNN

### 4.1 Dense MLP (SwiGLU)

The standard Llama-style MLP implements SwiGLU:

```
SwiGLU MLP:

  Input x --+-- FF1(x) = x @ W1 --> SiLU --+
            |                                +--> element-wise multiply --> FF2 --> Output
            +-- FF3(x) = x @ W3 ------------+

  Shapes (Llama3-70b on T3K, 8 devices, decode):
  x:    [1, 1, batch=32, dim=8192]        replicated, L1 width-sharded
  W1:   [8192, 28672/8] per device         column-parallel, DRAM-sharded
  W3:   [8192, 28672/8] per device         column-parallel, DRAM-sharded
  W2:   [28672/8, 8192] per device         row-parallel, DRAM-sharded
```

Data flow on T3K (1D Column+Row):
```
Step 1: FF1+FF3 (Column Parallel) - each device computes width shard
Step 2: SiLU + multiply (fused, per-device, L1) - ttnn.multiply with SiLU activation
Step 3: FF2 (Row Parallel) - produces [batch, 8192] PARTIAL per device
Step 4: ReduceScatter - sum 8 partials, distribute shards (or AllReduce)
```

Key implementation details:
- FF1 and FF3 use `ttnn.linear` with DRAM-sharded matmul in decode, 2D matmul in prefill
- SiLU activation is **fused** into the multiply: `ttnn.multiply(w1_out, w3_out, activation=SiLU)`
- FF2 output undergoes ReduceScatter on 1D meshes, AllReduce on 2D meshes
- For long prefill sequences (>= 1024), input is reshaped into smaller chunks

### 4.2 Mixture-of-Experts (MoE) Architecture

MoE replaces the single dense FFN with many expert FFNs. A router selects K experts per token from a pool of E experts.

```
MoE Architecture (e.g., GLM4-MoE-Lite: 64 experts, K=4, T3K):

  Input x --> Router (w_gate) --> Top-K selection (K=4 of 64)
                                        |
                 +----------------------+
                 v                      v
            topk_indices [T, 4]    topk_weights [T, 4]
            (which experts)        (how much each contributes)
                 |
                 v
          +---- Expert FFN Computation ----+
          | Only K out of E experts run    |
          | per token (sparse!)            |
          |                                |
          | D0: experts 0-7   (W1,W2,W3)  |
          | D1: experts 8-15  (W1,W2,W3)  |
          | D2: experts 16-23 (W1,W2,W3)  |
          | ...                            |
          | D7: experts 56-63 (W1,W2,W3)  |
          +--------------------------------+
                 |
                 v
          Weighted sum of expert outputs --> MoE output
```

### 4.3 Tensor Parallel vs Expert Parallel

```
+----------------------------------------------------------------------+
|              TENSOR PARALLEL (Dense MLP)                             |
|                                                                      |
|  Every device computes a SHARD of the single FFN.                   |
|  Each device has:  W_full[K/D, N]  (weight shard)                   |
|  All devices always active for every token.                          |
|  CCL: ReduceScatter or AllReduce to combine partial sums.           |
+----------------------------------------------------------------------+
|              EXPERT PARALLEL (MoE)                                   |
|                                                                      |
|  Each device stores FULL weights for E/D experts.                   |
|  D0: W1[0:8], W2[0:8], W3[0:8]  (experts 0-7, full weights)       |
|  D1: W1[8:16], W2[8:16], W3[8:16]  (experts 8-15)                 |
|  Devices with NO active experts skip computation entirely.           |
|  CCL: AllReduce (reduce path) or all_to_all (a2a path).            |
+----------------------------------------------------------------------+
```

### 4.4 The MoE-Specific Ops

#### 4.4.1 ttnn.scatter — Sparse-to-Dense Conversion

The router produces sparse (index, weight) pairs. Downstream ops need a dense 64-wide vector.

```
Input:  topk_indices = [3, 7, 12, 55]    topk_weights = [0.4, 0.3, 0.2, 0.1]
Base:   zeros [1, 1, T, 64]

         ==> scatter(base, dim=3, indices, values)

Output: [0,0,0,0.4,0,0,0,0.3,0,0,0,0,0.2,...,0,0.1,...,0]
                 ^           ^            ^           ^
              idx 3       idx 7       idx 12      idx 55
```

Uses ROW_MAJOR layout (indexed writes need contiguous memory). The zero tensor is cached because trace mode forbids allocation.

#### 4.4.2 ttnn.moe_expert_token_remap — Global to Local Routing

Takes 64-wide dense routing weights and produces per-device local information:

```
Input:  topk_weights_dense [1,1,T,64]  (global)
        expert_mapping [1,1,64,8]      (expert->device map)

         ==> moe_expert_token_remap

Output 1: local_weights [1,1,T,8]  routing weights for THIS device's 8 experts
Output 2: sparsity [T/32, 8]       block-level activity mask

Example for Device 0 (experts 0-7), token routed to expert 3:
  local_weights[t0] = [0, 0, 0, 0.4, 0, 0, 0, 0]
  sparsity block active for expert 3

Example for Device 4 (experts 32-39), token NOT routed here:
  local_weights[t0] = [0, 0, 0, 0, 0, 0, 0, 0]
  sparsity = all zeros -> entire block skipped
```

#### 4.4.3 ttnn.sparse_matmul — Skip-Zero-Blocks Matmul

The core compute primitive for EP MoE. A batched matmul where batch = experts, with a sparsity mask that skips inactive (block, expert) pairs.

```
Shapes:
  Input A (tokens):   [1, num_blocks, 32, hidden_size]
  Weight B (experts): [8, 1, hidden_size, intermediate_size]
  Sparsity mask:      [num_blocks, 8]
  Output:             [num_blocks, 8, 32, intermediate_size]

Hardware behavior (per block,expert pair):
  if sparsity[block][expert] == 0:
    SKIP: no DRAM read, no compute, no output write
  else:
    COMPUTE: full matmul for this block x expert

For decode (T=1, K=4, 8 local experts per device):
  At most 4/8 local experts active -> ~50% compute skipped
  Devices with no active experts -> 100% skipped
```

---

## Part 5: Expert-Parallel MoE — The Full Picture

### 5.1 Path A: Replicated-Token + AllReduce ("reduce")

This is the production path used with vLLM. Tokens are replicated on all devices.

```
REPLICATED-TOKEN PATH ("reduce"):

  Step 1: Input tokens REPLICATED on all 8 devices
          [1,1,T,2048] identical on D0..D7

  Step 2: Router IDENTICAL on every device (same input, same weights)
          x @ w_gate -> sigmoid -> topk(K=4)
          topk_weights [1,1,T,4], topk_indices [1,1,T,4]

  Step 3: scatter - sparse to dense (identical on every device)
          [1,1,T,4] -> [1,1,T,64] dense weight vector

  Step 4: moe_expert_token_remap - DIFFERENT per device
          D0: extracts local experts 0-7
          D1: extracts local experts 8-15 ...
          -> local_weights [1,1,T,8] + sparsity [T/32, 8]

  Step 5: sparse_matmul x 3 - SwiGLU FFN, DIFFERENT per device
          tokens x W1 -> gate;  tokens x W3 -> up
          SiLU(gate) * up -> x_ff;  x_ff x W2 -> output

  Step 6: Local aggregation - weight x sum across 8 local experts
          Each device has a PARTIAL result

  Step 7: <<< ALL_REDUCE >>> THE ONLY CCL OP
          Sum 8 devices partials over Ethernet
          Every device gets the COMPLETE MoE output

  Step 8: Output [1,1,T,2048] complete on all devices

  Total CCL cost: 1 AllReduce
```

### 5.2 Path B: All-to-All Dispatch + Combine ("a2a")

Tokens are sharded across devices (each device has T/D tokens).

```
ALL-TO-ALL PATH ("a2a"):

  Step 1: Input tokens SHARDED across devices
          D0: [1,1,T/8,2048]   D1: [1,1,T/8,2048]  ...

  Step 2: Router - each device routes its LOCAL tokens
          D0: topk for tokens 0..T/8

  Step 3: <<< ALL_TO_ALL_DISPATCH >>> CCL OP #1
          Send tokens to device(s) owning selected experts
          Returns dispatch_metadata

  Step 4: moe_expert_token_remap - build sparsity for received tokens

  Step 5: sparse_matmul x 3 - SwiGLU FFN on received tokens

  Step 6: <<< ALL_TO_ALL_COMBINE >>> CCL OP #2
          Send results back to originating devices

  Step 7: Weight x sum - apply routing weights

  Step 8: <<< ALL_REDUCE (other axis) >>> CCL OP #3 (optional, 2D only)

  Step 9: Output on originating devices

  Total CCL cost: 2-3 ops
```

### 5.3 Path Comparison

| Aspect | Reduce Path | A2A Path |
|--------|------------|----------|
| Token distribution | Replicated (8x) | Sharded (1x) |
| Router compute | Redundant (8x identical) | No redundancy |
| Expert FFN | Local only (no redundancy) | Local only |
| CCL ops | 1 (AllReduce) | 2-3 (dispatch+combine) |
| Decode (T=1) perf | Better (1 chain pass) | Worse (2 chain passes) |
| Memory efficiency | 8x token replication | 1x + inflation after dispatch |
| vLLM integration | Direct (already replicated) | Needs extra sharding |
| Code complexity | Simpler | More complex (metadata) |
| Best for | Production decode (vLLM) | Future data-parallel setups |

### 5.4 Complete Op Pipeline Side-by-Side

| Step | Reduce Path | A2A Path |
|------|------------|----------|
| 1. Distribute | Already replicated by vLLM | Sharded across devices |
| 2. Route | moe_topk_tt (identical on all) | moe_topk_tt (each routes its shard) |
| 3. Token shuffle | None | all_to_all_dispatch |
| 4. Sparse to Dense | scatter (identical everywhere) | From dispatch metadata |
| 5. Global to Local | moe_expert_token_remap -> weights+sparsity | moe_expert_token_remap -> sparsity |
| 6. Expert FFN | sparse_matmul x 3 (SwiGLU) | sparse_matmul x 3 (SwiGLU) |
| 7. Local aggregate | weight x sum | N/A |
| 8. Cross-device | all_reduce | all_to_all_combine |
| 9. Final reduce | N/A | all_reduce on other axis (if 2D) |
| **Total CCL** | **1** | **2-3** |

---

## Part 6: Serving Integration — Continuous Batching, vLLM, and Tracing

### 6.1 Continuous Batching

Traditional batching waits for batch_size requests, processes them all, then accepts new requests. Continuous batching slots new requests in as soon as a slot is free:

```python
while True:
    if not is_full(current_batch) and not prefill_q.empty():
        model_prefill(prefill_q.pop())   # prefill new request into free slot
    elif not is_empty(current_batch):
        model_decode(current_batch)      # decode step for all active users
    else:
        break
```

Benefits: reduces TTFT (time to first token), increases throughput by keeping batch full.

Requirements on the model:
- Must support single-user prefill (fill one slot at a time)
- Must support batched decode with per-user position IDs
- Must use paged KV cache for efficient memory management

### 6.2 vLLM Integration

Tenstorrent maintains a fork of vLLM for production serving on TT hardware. The model must implement a specific interface:

| Function | Purpose |
|----------|---------|
| `initialize_vllm_model` | Create model instance with mesh_device, batch_size, DP config |
| `allocate_kv_cache` | Return paged KV cache tensors |
| `prefill_forward` | Process prompt tokens, fill KV cache, return first decoded token |
| `decode_forward` | Generate next token for all active users (traced, padded to max_batch) |
| `warmup_model_prefill` | Compile and capture traces for prefill |

Key constraints:
- Must use paged attention ops (paged_fill_cache, paged_update_cache, paged_sdpa_decode)
- decode_forward tokens padded to max_batch_size for tracing compatibility
- Tokens arrive replicated across devices, matching the EP reduce path

Backend files: `tt.py` (platform), `tt_loader.py` (model init), `tt_worker.py` (device/KV), `tt_model_runner.py` (execution)

### 6.3 Tracing

Tracing records a single pass and replays it as one device command, eliminating host dispatch overhead:

```
Without Tracing:                    With Tracing:
  Host dispatches each OP             Single replay command
  OP-to-OP gap: 100s of us            OP-to-OP gap: <6 us
```

Critical limitation: **No tensor allocation/deallocation during a trace.** Every buffer must have the same size every replay. This is why:
- Decode uses tracing (fixed shapes: batch_size x 1 token)
- Prefill does NOT use tracing (variable sequence lengths)
- vLLM pads decode inputs to max_batch_size
- MoE scatter uses a cached zero tensor

### 6.4 How vLLM, Continuous Batching, and EP MoE Connect

```
vLLM Server:

  Incoming requests -> Scheduler -> Continuous Batching
      |
      v (for each new request)
  1. Find free slot in decode batch
  2. Prefill: tokens -> paged_fill_cache
  3. Insert into decode batch
      |
      v (each decode step)
  decode_forward(all_active_tokens, ...)
      |
      +-- Inside model (traced):
      |   +-- Embedding
      |   +-- N x Decoder layers:
      |   |   +-- Attn norm -> Attention (TP col/row, paged SDPA)
      |   |   +-- Residual add
      |   |   +-- FFN norm -> MLP or MoE:
      |   |   |   Dense: TP col+row, ReduceScatter
      |   |   |   MoE: EP reduce, router->scatter->remap->sparse_matmul->AllReduce
      |   |   +-- Residual add
      |   +-- Final norm -> LM Head
      |
      v
  Output -> Sampler -> Check EOS -> Update/Free slots
```

---

## Part 7: Performance Optimization Guide

### 7.1 The Five Performance Components

1. **Main Python Thread** — your code dispatching ttnn calls. Mitigate: avoid torch.nn.Module, precompute configs, profile with viztracer.
2. **Host API** — C++ processing (generally out of your control).
3. **Host-Device Communications** — PCIe, tilize/untilize. Mitigate: embed on-device, sample on-device, minimize transfers.
4. **Device Dispatch (OP-to-OP gap)** — time between OPs. Mitigate: TRACING (<6us gap), fuse OPs.
5. **Device OP Performance** — actual compute. Mitigate: DRAM-sharded matmul, L1 sharding, precision tuning.

### 7.2 Matmul Variant Selection

| Scenario | Variant | Why |
|----------|---------|-----|
| Decode, DRAM-BW-bound | DRAM-Sharded Matmul | Weights sharded across 12 DRAM banks, ~240 GB/s |
| Prefill, compute-bound | Matmul 2D | Parallelize over M and N, DRAM interleaved |
| Small decode matmuls | Matmul 1D | Parallelize over N only, width-sharded L1 |

### 7.3 Compute Kernel Configuration

| Fidelity | Speed | When to use |
|----------|-------|------------|
| HiFi4 | 1x | BF16 weights, attention ops |
| HiFi2 | 2x | BFP8 weights (standard MLP) |
| LoFi | 3.6x | BFP4 weights (aggressive) |

```python
compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

### 7.4 CCL Optimization

- Use Column+Row parallel for MLP to defer CCL past dimension expansion (3.5x less data)
- Use BFP8 inputs to CCL ops (halves data volume)
- Use AllGatherMatmul fusion: `ttnn.experimental.all_gather_matmul` overlaps communication with compute
- For 2D meshes: reduce along axis with fewer devices first

### 7.5 MoE-Specific Performance

1. **Sparsity exploitation**: sparse_matmul skips inactive (block, expert) pairs. For decode (T=1, K=4, 8 local experts): ~50% skip. Devices with no active experts: 100% skip.
2. **Memory**: Expert weights sharded (no redundancy). Tokens replicated in reduce path (8x, acceptable for decode).
3. **Path selection**: Decode -> always reduce (1 CCL). Sharded tokens -> consider a2a.
4. **Trace compatibility**: Scatter zero tensor cached. All buffer sizes fixed.

### 7.6 General Performance Tips

- Use as many cores as possible
- Move data as little as possible
- Avoid unnecessary ShardedToInterleaved / InterleavedToSharded conversions
- Always use ScaledDotProductAttention (SDPA) ops for attention
- For matmuls: output subblock >= 2x1, in0_block_w >= 2, use lowest precision that works
- Do NOT recreate config objects every forward pass
- Do NOT subclass torch.nn.Module for hot paths
- Generate shard specs and kernel configs once in the constructor

---

## Appendices

### Appendix A: Decoder Layer Architecture

```
              +-------------+
              |   Input x   |
              +------+------+
                     |
              +------v------+
              |  Attn Norm  |  (RMSNorm, distributed on multi-device)
              +------+------+
                     |
              +------v------+
              |  Attention  |  QKV proj -> RoPE -> KV cache -> SDPA -> out proj
              |  (TP: col+  |  CCL: AllGather or ReduceScatter
              |   row par.) |
              +------+------+
                     |
              +------v------+
              | Residual Add|  x + attn_out
              +------+------+
                     |
              +------v------+
              |  FFN Norm   |
              +------+------+
                     |
         +-----------v-----------+
         |     Dense MLP         |        MoE Layer
         |  FF1+FF3 (col-par)   |   Router -> topK
         |  SiLU(FF1) * FF3     |   scatter + remap
         |  FF2 (row-par)       |   sparse_matmul x 3
         |  ReduceScatter       |   AllReduce
         +-----------+-----------+
                     |
              +------v------+
              | Residual Add|  h + ff_out
              +------+------+
                     |
              +------v------+
              |   Output    |
              +-------------+
```

### Appendix B: LM Head

For large vocabularies (128K), weights are split across devices AND iterations:

```
LM Head (Llama3.1: vocab=128K, dim=8K):
  Weight [8K, 128K] -> split across devices + chunks
  Forward: iterate over weight chunks, DRAM-sharded matmul each, concat
  Activation must be replicated before LM Head
```

### Appendix C: Prefill vs Decode Quick Reference

| Aspect | Prefill | Decode |
|--------|---------|--------|
| Purpose | Process full prompt, fill KV cache | Generate one token at a time |
| Parallelization | Over sequence length | Over batch (users) |
| Input shape | [1, 1, seq_len, dim] | [1, 1, batch, dim] |
| Memory | DRAM interleaved | L1 width-sharded |
| Matmul variant | 2D (compute-bound) | DRAM-Sharded (BW-bound) |
| Tracing | NO (variable seq_len) | YES (fixed shapes) |
| KV cache | paged_fill_cache (bulk) | paged_update_cache (single token) |
| Attention | scaled_dot_product_attention | scaled_dot_product_attention_decode |
| Long seq | Reshape into smaller chunks | Not needed (seq_len=1) |
| LM Head | Only on last tile | On full batch |

### Appendix D: Performance Metrics

| Metric | Formula | Optimized by |
|--------|---------|-------------|
| TTFT (Time to First Token) | Prefill latency | Prefill throughput, batch=1 |
| Total Throughput | batch_size / decode_step_latency | Increasing batch, decode speed |
| User Throughput | 1 / decode_step_latency | Decreasing batch, decode speed |

---

*This document synthesizes the official TT-Metal LLM tech report, the TTNN CCL/EP-MoE analysis, and the vLLM integration guide into a unified reference for understanding LLM implementations on Tenstorrent hardware.*
