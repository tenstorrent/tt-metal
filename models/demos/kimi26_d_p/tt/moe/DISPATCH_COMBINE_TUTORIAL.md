# MoE Dispatch and Combine: A Tutorial

This tutorial explains, from first principles, how the dispatch and combine
stages of this DeepSeek-V3 Mixture-of-Experts implementation work on
Tenstorrent multi-chip hardware.  It assumes you understand how DeepSeek-V3's
MoE routing works (gate, top-k selection, weighted aggregation) but have
**no prior experience** with collective communication libraries (CCLs), device
meshes, or fabric interconnects.

---

## Table of Contents

1. [The Problem: Why Do We Need Dispatch and Combine?](#1-the-problem-why-do-we-need-dispatch-and-combine)
2. [Hardware Primer: Chips, Meshes, and Communication](#2-hardware-primer-chips-meshes-and-communication)
3. [Expert Placement: Which Expert Lives Where?](#3-expert-placement-which-expert-lives-where)
4. [The Expert Dispatch Table](#4-the-expert-dispatch-table)
5. [Routing Setup: Computing Where to Write](#5-routing-setup-computing-where-to-write)
6. [The Dispatch Buffer: Memory Layout](#6-the-dispatch-buffer-memory-layout)
7. [Dispatch: Moving Tokens to Experts](#7-dispatch-moving-tokens-to-experts)
8. [Combine: Moving Results Back](#8-combine-moving-results-back)
9. [Post-Combine Reduce: Weighted Aggregation](#9-post-combine-reduce-weighted-aggregation)
10. [Worked Example: 2 Chips, 16 Experts](#10-worked-example-2-chips-16-experts)
11. [Scaling to 2D Meshes](#11-scaling-to-2d-meshes)
12. [CCL Glossary](#12-ccl-glossary)
13. [Source File Map](#13-source-file-map)

---

## 1. The Problem: Why Do We Need Dispatch and Combine?

In DeepSeek-V3, each MoE layer has 256 routed experts. Each token is
sent to its top-8 experts. On a single GPU this is conceptually simple: select
the experts, run each expert's FFN on its tokens, weight the outputs, sum
them up.

On multi-chip hardware, the 256 experts are **distributed** across devices
because no single chip has enough memory to hold all their weights. This means
a token sitting on chip 0 might need to be processed by an expert whose weights
live on chip 3. We need to physically **move** that token's embedding across
chips.

That is the dispatch/combine problem:

- **Dispatch**: After the gate picks each token's top-k experts, send every
  token embedding to the chip that hosts each of its chosen experts.
- **Combine**: After each expert processes its tokens, send the results back
  to the chip where the token originated and reassemble the outputs.

Together, dispatch and combine implement an *all-to-all* exchange: every chip
can send tokens to every other chip, and receive results back.

### Why not just replicate experts?

Replicating all 256 experts on every chip would eliminate the need for
communication, but each expert's FFN has three weight matrices
(`gate_proj`, `up_proj`, `down_proj`) totaling tens of megabytes. 256 experts
would require tens of gigabytes of weight storage on *every* chip. This is
infeasible given the on-device memory budget.

### Why not use sparse padding?

A naive alternative is to give every expert a slot for every token and fill
unused slots with zeros. But if you have 4096 tokens per chip and 256 experts,
that is a 4096 x 256 grid where each token only uses 8 of the 256 slots. The
resulting matrix is 97% zeros. Multiplying an expert's weights against a
mostly-zero input wastes 97% of the compute.

This implementation uses **dense packing**: only the tokens that are actually
routed to a given expert are placed in that expert's buffer. The expert's
matmul then has zero wasted compute.

---

## 2. Hardware Primer: Chips, Meshes, and Communication

### What is a mesh?

A Tenstorrent system consists of multiple chips arranged in a **mesh** — a 2D
grid with rows and columns. A mesh is described by its shape `(rows, cols)`.

```
Example: MeshDevice(rows=4, cols=2) — an 8-chip system

   Column 0          Column 1
┌──────────────┬──────────────┐
│  Chip (0,0)  │  Chip (0,1)  │  ← Row 0
├──────────────┼──────────────┤
│  Chip (1,0)  │  Chip (1,1)  │  ← Row 1
├──────────────┼──────────────┤
│  Chip (2,0)  │  Chip (2,1)  │  ← Row 2
├──────────────┼──────────────┤
│  Chip (3,0)  │  Chip (3,1)  │  ← Row 3
└──────────────┴──────────────┘
```

Each chip has its own local DRAM, L1 SRAM, and compute cores.

### How chips communicate: NOC and fabric

There are two mechanisms for moving data between locations:

- **NOC (Network on Chip)**: moves data between cores and memory *within a
  single chip*. Very fast, very low latency. Used when a token's destination
  expert happens to be on the same chip.

- **Fabric**: connects chips to each other via ethernet links. Used when a
  token must travel to a different chip. Each chip-to-chip connection can have
  one or more **links** (physical ethernet lanes). More links = more bandwidth.

The dispatch and combine kernels abstract this away: for each token, they check
whether the destination is local (use NOC) or remote (use fabric), and pick
the right mechanism automatically.

### What is a CCL?

A **Collective Communication Library (CCL)** provides high-level operations
that coordinate data movement across multiple chips. Instead of manually
programming each chip-to-chip transfer, you call a CCL operation and it
handles the coordination.

The CCL operations used in this implementation are:

| CCL Operation | What It Does | Where It's Used |
|---|---|---|
| **all-gather** | Every chip sends its local data to every other chip. Afterwards, all chips have the complete concatenated data. | Routing setup (share per-chip histograms), TP gather of input embeddings |
| **reduce-scatter** | Every chip contributes data; the results are summed element-wise and then split (scattered) so each chip gets a portion of the reduced result. | Post-combine reduction, shared expert output |
| **all-reduce** | Every chip contributes data; the results are summed and replicated so every chip gets the full sum. | Gate matmul partial products across TP |

### Topology: Linear vs Ring

When chips communicate along a mesh axis, they can be arranged in two
**topologies**:

- **Linear**: chip 0 → chip 1 → chip 2 → chip 3. Data flows in one direction.
  Simple but the endpoints only have one neighbor.
- **Ring**: chip 0 → chip 1 → chip 2 → chip 3 → chip 0. Data flows in a
  circle. Better bandwidth utilization because every chip has two neighbors.

This implementation uses `ttnn.Topology.Linear` by default.

### Cluster axis

In a 2D mesh, CCL operations happen along a specific **axis**:

- `cluster_axis=0` means "communicate along **rows**" (chips in the same
  column talk to each other). This is the **dispatch axis** — the axis along
  which tokens travel during dispatch/combine.
- `cluster_axis=1` means "communicate along **columns**" (chips in the same
  row talk to each other). This is the **TP axis** — used for tensor-parallel
  weight sharding and reduction.

---

## 3. Expert Placement: Which Expert Lives Where?

The 2D mesh is used for two kinds of parallelism simultaneously:

| Mesh Dimension | Name | Purpose |
|---|---|---|
| Rows (axis 0) | Sequence Parallel (SP) / Dispatch axis | Tokens are distributed across rows. Dispatch/combine all-to-all happens here. |
| Columns (axis 1) | Expert Parallel (EP) / Tensor Parallel (TP) axis | Each column is an independent **dispatch group** that hosts a subset of experts. |

### Dispatch groups

A **dispatch group** is a set of chips that collectively host a subset of
experts and exchange tokens among themselves. All the chips in one column of
the mesh form a dispatch group.

- `dispatch_group_size` = number of rows = how many chips are in each group.
- `num_dispatch_groups` = number of columns = how many independent groups
  exist.

Each dispatch group handles `num_routed_experts / num_dispatch_groups` experts.
Within a group, those experts are evenly divided across the chips:

```python
experts_per_group = num_routed_experts // num_dispatch_groups
experts_per_chip  = experts_per_group  // dispatch_group_size
```

### Concrete placement

For a `4×2` mesh with 16 experts:

```
experts_per_group = 16 / 2 = 8    (each column handles 8 experts)
experts_per_chip  = 8  / 4 = 2    (each chip hosts 2 experts)

Column 0 (Dispatch Group 0)     Column 1 (Dispatch Group 1)
┌────────────────────────┐     ┌────────────────────────┐
│ Chip (0,0): experts 0,1│     │ Chip (0,1): experts 8,9│
│ Chip (1,0): experts 2,3│     │ Chip (1,1): experts10,11│
│ Chip (2,0): experts 4,5│     │ Chip (2,1): experts12,13│
│ Chip (3,0): experts 6,7│     │ Chip (3,1): experts14,15│
└────────────────────────┘     └────────────────────────┘
```

### Why partition experts this way?

1. **Memory**: each chip only stores weights for its `experts_per_chip` experts.
   With 256 experts across 32 chips, each chip holds 8 experts — a manageable
   amount of weight memory.

2. **Communication scope**: dispatch/combine only needs to exchange tokens
   within a dispatch group (one column), not across the entire mesh. This keeps
   the communication pattern local.

3. **Independent groups**: the two columns operate independently. A token on
   chip (2,0) that needs expert 5 only communicates with chips in column 0 (its
   dispatch group). It never talks to chips in column 1.

---

## 4. The Expert Dispatch Table

Before any tokens move, the system builds a static lookup table that answers:
"given a global expert ID, which chip within my dispatch group hosts it?"

```python
expert_dispatch_table[group, expert_id] → chip_id  (or -1 if not in this group)
```

For the 4×2 mesh above, the table looks like:

```
Group 0: [ 0, 0, 1, 1, 2, 2, 3, 3, -1,-1,-1,-1,-1,-1,-1,-1]
Group 1: [-1,-1,-1,-1,-1,-1,-1,-1,  0, 0, 1, 1, 2, 2, 3, 3]
          ├── experts 0-7 ──────┤  ├── experts 8-15 ────────┤
```

Reading this: in Group 0, expert 0 is on chip 0, expert 1 is also on chip 0,
expert 2 is on chip 1, and so on. Experts 8-15 show `-1` because they don't
exist in Group 0 — they belong to Group 1.

### How the table is sharded across devices

This table has shape `(num_dispatch_groups, num_routed_experts)`. It is
distributed to devices using a **mesh mapper**:

- **Replicated** across the dispatch axis (rows): every chip in the same
  dispatch group needs the same row of the table.
- **Sharded** across dispatch groups (columns): each column only needs its
  own row.

After sharding, every device holds `(1, num_routed_experts)` — a single row
of the table corresponding to its own dispatch group.

### What is a mesh mapper?

A `ShardTensor2dMesh` is a TTNN helper that distributes a host tensor across a
2D mesh of devices. You specify `dims=(axis0_dim, axis1_dim)`:

- A numeric value means "shard this tensor dimension across this mesh axis"
  (split the tensor into pieces, one per device along that axis).
- `None` means "replicate along this mesh axis" (every device along that axis
  gets a full copy).

For example, `dims=(None, 0)` means: replicate across mesh rows, shard tensor
dimension 0 across mesh columns.

---

## 5. Routing Setup: Computing Where to Write

After the gate selects each token's top-k experts, the system needs to
figure out *exactly where* in the destination chip's memory buffer each token
should be written. This is the job of `TtMoERoutingSetup`, and it involves
two device operations run in sequence.

### Step 1: masked_bincount — "How many tokens go to each expert from my chip?"

Each chip independently counts how many of its tokens are routed to each
expert. This is a histogram:

```
Chip 0's histogram: [12, 8, 15, 5, ...]  (12 tokens → expert 0, 8 → expert 1, ...)
Chip 1's histogram: [ 9, 11, 7, 14, ...]
```

The "masked" part means the kernel uses the dispatch table to skip experts
that don't belong to this chip's dispatch group (their count stays 0).

**Implementation detail**: the histogram is computed in parallel across 64
cores (8×8 grid). The token indices are height-sharded so each core processes
`seq_len / 64` tokens. A binary-tree reduction combines per-core counts into
a single histogram.

### Step 2: offset_cumsum — "What is my write position in the destination buffer?"

This is the most nuanced step. It computes a **global write offset** for each
`(source_chip, expert)` pair — the exact token index in the destination chip's
dispatch buffer where this source chip should start writing its tokens for that
expert.

#### First: all-gather the histograms

Each chip only knows its own histogram. To compute offsets, every chip needs
to know *every other chip's* histogram. An **all-gather** along axis 0
(the dispatch axis) concatenates all histograms:

```
Before all-gather:            After all-gather (on every chip):
  Chip 0: [12, 8, 15, ...]     gathered: [[12, 8, 15, ...],   ← chip 0's counts
  Chip 1: [ 9,11,  7, ...]                [ 9,11,  7, ...],   ← chip 1's counts
  Chip 2: [ 6, 3, 10, ...]                [ 6, 3, 10, ...],   ← chip 2's counts
  Chip 3: [ 4, 7,  2, ...]]               [ 4, 7,  2, ...]]   ← chip 3's counts
```

#### Then: compute the two offset components

The global offset for source chip `k` and expert `e` is the sum of two parts:

**Component 1 — Local offset** (different per source chip):

This ensures that when multiple source chips send tokens to the same expert,
their writes don't overlap. It is a *prefix sum* over the histogram:

```
local_offset_0[e] = 0                                    (chip 0 writes first)
local_offset_1[e] = histogram[0, e]                      (chip 1 writes after chip 0)
local_offset_2[e] = histogram[0, e] + histogram[1, e]    (after chips 0 and 1)
...
```

For expert 0 with counts [12, 9, 6, 4]:
- Chip 0 starts at offset 0
- Chip 1 starts at offset 12
- Chip 2 starts at offset 21
- Chip 3 starts at offset 27

**Component 2 — Expert region offset** (same for all source chips):

A single destination chip may host multiple experts (e.g., experts 0 and 1
both live on chip 0). Their token regions must not overlap within the chip's
buffer. The region offset tells each expert where its region starts in the
flat buffer:

```
Expert 0's region starts at token index 0
Expert 1's region starts at token index ceil(total_count[0] / 32) * 32
```

The `TILE_HEIGHT` (32) alignment ensures each region starts on a tile
boundary. This matters because Tenstorrent hardware processes data in tiles
(32×32 blocks). Starting at a non-aligned offset would require expensive
partial-tile handling.

**Final formula**:

```
global_offset[chip_k, expert_e] = local_offset_k[e] + expert_region_offset[e]
```

This single number tells chip `k` exactly where in the destination chip's
buffer to start writing its tokens for expert `e`.

---

## 6. The Dispatch Buffer: Memory Layout

Each chip allocates a **flat dispatch buffer** in DRAM:

```
Shape: (1, 1, max_dispatch_buffer_token_size, emb_dim)
```

This buffer is *shared* across all experts hosted on that chip. The experts'
token regions are packed contiguously within it:

```
Chip 0's dispatch buffer (hosts experts 0 and 1):

Token index:   0                31|32               63|64  ...
              ├── Expert 0's region ──┤── Expert 1's region ──┤
              │ chip0's tokens │ chip1's│ chip0's    │ chip1's │
              │ for expert 0   │ tokens │ for exp 1  │ tokens  │
              └────────────────┘───────┘─────────────┘─────────┘
                  ↑ tile-aligned    ↑ tile-aligned
```

Within each expert's region, tokens from different source chips are packed
in order: all of chip 0's tokens first, then chip 1's, then chip 2's, etc.
The local offset ensures there are no gaps or overlaps.

### Why flat instead of multi-dimensional?

A logical layout would be `[experts_per_chip, max_tokens_per_expert, emb_dim]`.
The flat layout is chosen because:

1. **Dynamic token counts**: expert 0 might receive 50 tokens while expert 1
   receives 200. A fixed `max_tokens_per_expert` per expert would waste memory
   on the expert that got fewer tokens.

2. **Tile alignment**: by using region offsets with TILE_HEIGHT alignment, each
   expert's data starts at a tile boundary without requiring an extra dimension.

3. **Single contiguous allocation**: one big buffer per chip is simpler to
   manage than multiple smaller buffers, and remote fabric writes can address
   any position in it.

### Buffer sizing

```python
max_dispatched_tokens_per_expert = dispatch_group_size * seq_len_per_chip
max_dispatch_buffer_token_size = max_dispatched_tokens_per_expert * capacity_factor
```

`max_dispatched_tokens_per_expert` is the theoretical worst case: every token
from every chip in the dispatch group routes to the same expert.
`capacity_factor` scales this up to account for multiple experts sharing the
buffer. Tokens that would overflow the buffer are silently dropped.

### Metadata buffer

A parallel buffer with the same token dimension stores 5 metadata fields per
token:

```
Shape: (1, 1, max_dispatch_buffer_token_size, 5)

Fields:
  [0] linearized_mesh_coord  — which source chip sent this token
  [1] token_idx              — the token's original index in the source sequence
  [2] topk_idx               — which of the token's top-k expert slots this is
  [3] routed_expert          — the global expert ID
  [4] weight                 — the router weight (bfloat16 stored as int32 bits)
```

This metadata is the "return address" that lets the combine stage send results
back to the right place. It is written by dispatch and read by combine.

---

## 7. Dispatch: Moving Tokens to Experts

With offsets computed and buffers allocated, dispatch performs the actual data
movement. Here is the per-token logic:

```
for each token on this source chip:
    for each of the token's top-k expert choices:
        expert_id = indices[token, topk_slot]
        dest_chip = expert_dispatch_table[expert_id]

        if dest_chip == -1:
            skip  (expert not in my dispatch group)

        write_position = global_offset[my_chip, expert_id]
        global_offset[my_chip, expert_id] += 1

        if dest_chip == my_chip:
            # Expert is local — write via NOC (fast, on-chip)
            dispatch_buffer[write_position, :] = token_embedding
            metadata[write_position, :] = (my_coord, token_idx, topk_slot, expert_id, weight)
        else:
            # Expert is remote — write via fabric (cross-chip ethernet)
            remote_chip[dest_chip].dispatch_buffer[write_position, :] = token_embedding
            remote_chip[dest_chip].metadata[write_position, :] = (my_coord, ...)
```

### Key properties of dispatch

1. **Write-only, no reads**: the source chip writes into the destination chip's
   buffer. It does not read from the destination. This is important because
   fabric writes are much more efficient than reads on this hardware.

2. **No synchronization needed for non-overlapping offsets**: because the
   routing setup computed non-overlapping offsets for each source chip, multiple
   source chips can write to the same destination chip's buffer simultaneously
   without data races.

3. **Self-contained per chip**: each chip runs the dispatch kernel independently.
   There is no global barrier or coordinator. The offsets encode all the
   coordination.

### The TTNN wrapper

The Python code is a thin wrapper. All the real work happens in the C++
device kernel:

```python
(dispatched_buffer, metadata) = ttnn.experimental.deepseek_prefill.dispatch(
    input_tensor=x,
    weights_tensor=weights,
    indices_tensor=indices,
    expert_offsets_tensor=tt_expert_offsets,
    expert_dispatch_table_tensor=tt_expert_dispatch_table,
    dispatch_group_size=...,
    experts_per_chip=...,
    num_routed_experts=...,
    num_experts_per_tok=...,
    metadata_len=5,
    max_dispatch_buffer_token_size=...,
    cluster_axis=0,        # communicate along mesh rows
    num_links=1,           # 1 ethernet link per connection
    topology=Linear,       # linear chain topology
)
```

---

## 8. Combine: Moving Results Back

After dispatch, each chip's buffer contains tokens for its local experts.
The routed expert FFN processes them in place. Then combine reverses the
token movement: it sends each expert's output back to the chip where the token
originally came from.

### The reverse lookup via metadata

Combine reads the metadata that dispatch wrote alongside each token:

```
for each expert on this chip:
    for each valid token slot (bounded by expert_token_counts):
        (src_chip, token_idx, topk_idx, expert_id, weight) = metadata[slot]

        weighted_output = expert_output[slot, :] * weight

        if src_chip == my_chip:
            # Token originated here — write via NOC
            output[token_idx, topk_idx, :] = weighted_output
        else:
            # Token came from another chip — write via fabric
            remote_chip[src_chip].output[token_idx, topk_idx, :] = weighted_output
```

### Output buffer shape

```
Per device: (1, 1, seq_len_per_chip, num_experts_per_tok, emb_dim)
```

This is a **token-centric** layout — the original ordering is restored. For
each token, there are `num_experts_per_tok` (e.g. 8) slots, one for each
expert the token was routed to. The combine kernel fills each slot with the
weighted output from that expert.

### Sparsity in the output

After combine, the output on each chip is **sparse**. A chip in dispatch
group 0 only has valid data in the top-k slots corresponding to experts 0-7
(its group's experts). Slots for experts 8-15 are zeros (from the
`init_zeros=True` initialization). This sparsity is resolved by the reduce
step.

### The TTNN wrapper

```python
output = ttnn.experimental.deepseek_prefill.combine(
    dispatched_buffer,       # expert outputs (same buffer, overwritten in-place)
    dispatched_metadata,     # metadata from dispatch (unchanged)
    expert_token_counts,     # how many valid tokens per expert
    expert_region_offsets,   # where each expert's region starts
    dispatch_group_size=...,
    experts_per_chip=...,
    num_experts_per_tok=...,
    seq_len_per_chip=...,
    cluster_axis=0,
    num_links=1,
    topology=Linear,
    init_zeros=True,         # zero-initialize output before writing
)
```

---

## 9. Post-Combine Reduce: Weighted Aggregation

After combine, each chip has a sparse `(seq_len, topk, emb_dim)` tensor.
The `TtReduceModule` collapses this into the final per-token output in two
steps:

### Step 1: Fused weighted sum over top-k

For each token, sum the expert contributions weighted by the router scores:

```
routed_output[token, :] = sum over k: scores[token, k] * combine_output[token, k, :]
```

This is done by a fused TTNN kernel (`post_combine_reduce`) that:

- Reads the sparse combine output in ROW_MAJOR layout
- Uses the dispatch table and indices to identify which top-k slots belong
  to local experts (skipping the rest — they're zeros anyway)
- Multiplies by weights and sums, writing TILE_LAYOUT output

On a TP-4 setup, each chip only has valid data for 25% of the top-k slots.
Skipping the other 75% saves significant compute.

### Step 2: Reduce-scatter across TP axis

After the weighted sum, each chip has a partial result (only its dispatch
group's expert contributions). To get the full result, the partial sums from
all dispatch groups must be added together.

A **reduce-scatter** along axis 1 (the TP/EP axis) does this:

1. **Reduce**: sum each chip's partial result element-wise across all columns.
2. **Scatter**: split the summed embedding dimension across chips.

Output per chip: `(seq_len, emb_dim / num_dispatch_groups)`

This matches the sharding of the shared expert output, so the two can be added
directly.

---

## 10. Worked Example: 2 Chips, 16 Experts

Let's trace through the full dispatch/combine flow with a small concrete
configuration:

```
Mesh: 2×1 (2 chips, 1 dispatch group)
seq_len_per_chip: 32
num_routed_experts: 16
num_experts_per_tok: 4
emb_dim: 7168
```

### Derived values

```
dispatch_group_size = 2
num_dispatch_groups = 1
experts_per_chip = 16 / 2 = 8
max_dispatched_tokens_per_expert = 2 * 32 = 64
```

### Expert placement

```
Chip 0: experts 0, 1, 2, 3, 4, 5, 6, 7
Chip 1: experts 8, 9, 10, 11, 12, 13, 14, 15
```

### Dispatch table

```
Group 0: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
```

### Suppose token 5 on chip 0 routes to experts [2, 9, 14, 0]

- Expert 2 → dispatch_table[2] = 0 → same chip, write via NOC
- Expert 9 → dispatch_table[9] = 1 → remote chip 1, write via fabric
- Expert 14 → dispatch_table[14] = 1 → remote chip 1, write via fabric
- Expert 0 → dispatch_table[0] = 0 → same chip, write via NOC

After dispatch:
- Chip 0's buffer has token 5's embedding in expert 2's region and expert 0's
  region.
- Chip 1's buffer has token 5's embedding in expert 9's region and expert 14's
  region.

### After routed experts process the tokens

Each chip runs its 8 local experts on the tokens in its buffer. The expert
outputs overwrite the buffer.

### Combine sends results back

Chip 0 reads metadata for experts 0 and 2, finds `src_chip=0, token_idx=5`,
writes results locally.

Chip 1 reads metadata for experts 9 and 14, finds `src_chip=0, token_idx=5`,
writes results back to chip 0 via fabric.

### Result on chip 0

```
output[5, 0, :] = expert_0_output * weight_0   (slot 0, local write)
output[5, 1, :] = expert_9_output * weight_1   (slot 1, received from chip 1)
output[5, 2, :] = expert_14_output * weight_2  (slot 2, received from chip 1)
output[5, 3, :] = expert_2_output * weight_3   (slot 3, local write)
```

The topk_idx field determines which slot each expert's output goes into,
preserving the original ordering from the gate.

---

## 11. Scaling to 2D Meshes

On a `4×2` mesh, two things change:

### Two independent dispatch groups

Column 0 and column 1 each form their own dispatch group. Token routing within
column 0 is completely independent from column 1. This doubles the expert
parallelism without increasing per-group communication.

### TP axis for shared expert + reduction

The column dimension (axis 1) adds a second communication axis. It is used
for:

1. **All-gather** of input embeddings: the input `x` arrives TP-sharded
   (each chip in a row has `emb_dim / num_cols` of the embedding). Before
   dispatch, an all-gather along axis 1 replicates the full embedding on
   every chip.

2. **Reduce-scatter** after combine: each dispatch group produces a partial
   result. The reduce-scatter along axis 1 sums across groups and shards the
   result.

3. **Shared expert** TP: the shared expert's weights are sharded across
   columns. Its output is reduce-scattered along axis 1 to match the routed
   output's sharding.

### The full CCL operation map

```
Axis 0 (rows, dispatch groups):
  - all-gather in routing setup (share histograms)
  - fabric writes in dispatch (token → expert)
  - fabric writes in combine (result → origin)

Axis 1 (columns, TP):
  - all-gather of input x (replicate full embedding)
  - all-reduce of gate logits (sum partial matmul products)
  - reduce-scatter of routed output (sum across groups, shard embedding)
  - reduce-scatter of shared expert output
```

---

## 12. CCL Glossary

| Term | Definition |
|---|---|
| **all-gather** | Each device contributes a piece; the result is the concatenation of all pieces, replicated on every device. |
| **all-reduce** | Each device contributes a tensor; results are summed element-wise and the full sum is replicated on every device. |
| **reduce-scatter** | Each device contributes a tensor; results are summed element-wise, then the sum is split (scattered) so each device gets one shard. Equivalent to all-reduce followed by chunking. |
| **cluster_axis** | Which mesh dimension (0=rows, 1=columns) a CCL operation communicates along. |
| **topology** | How devices are connected along an axis: `Linear` (chain) or `Ring` (loop). |
| **num_links** | Number of physical ethernet connections between adjacent devices. More links = more bandwidth. |
| **fabric** | The inter-chip interconnect (ethernet). Dispatch/combine use direct fabric writes rather than CCL collectives for the token data movement, because the write pattern is irregular (each token goes to a different destination). |
| **NOC** | Network on Chip — the intra-chip interconnect between cores and memory. Used when source and destination are on the same chip. |
| **mesh mapper** (`ShardTensor2dMesh`) | A TTNN helper that distributes a host tensor to devices on a 2D mesh. `dims=(a, b)`: shard tensor dim `a` across mesh rows, dim `b` across mesh cols. `None` = replicate. |
| **mesh composer** (`MeshComposerConfig`) | The inverse of a mesh mapper: reassembles per-device tensors back to a single host tensor. `dims=[a, b]`: concatenate mesh rows into tensor dim `a`, mesh cols into tensor dim `b`. |
| **dispatch group** | A set of chips that collectively host a subset of experts and exchange tokens among themselves. Corresponds to one column of the mesh. |
| **dispatch axis** | The mesh axis (rows, axis 0) along which chips within a dispatch group communicate. |
| **TP axis** | The mesh axis (columns, axis 1) used for tensor-parallel weight sharding and cross-group reduction. |
| **height sharding** | Splitting a tensor's rows across cores for parallel processing (e.g., `masked_bincount` uses 64 cores with each processing `seq_len / 64` rows). |
| **TILE_HEIGHT** | 32 rows — the native tile size on Tenstorrent hardware. Expert regions in the dispatch buffer are aligned to this boundary for efficient memory access. |
| **ROW_MAJOR / TILE_LAYOUT** | Data layout formats. ROW_MAJOR stores elements row-by-row (natural for irregular access patterns like dispatch). TILE_LAYOUT stores elements in 32×32 tiles (efficient for matmuls). Dispatch/combine work in ROW_MAJOR; expert FFNs work in TILE_LAYOUT. |

---

## 13. Source File Map

| File | Role |
|---|---|
| `tt_moe.py` | Top-level `TtMoe` module — assembles the full pipeline (gate → dispatch → experts → combine → reduce → add). |
| `tt_moe_routing_setup.py` | `TtMoERoutingSetup` — computes per-expert histograms (`masked_bincount`) and write offsets (`offset_cumsum`). |
| `tt_dispatch.py` | `TtDispatchModule` — thin wrapper around the dispatch device kernel. Contains sharding helpers for `expert_offsets` and `expert_dispatch_table`. |
| `tt_combine.py` | `TtCombineModule` — thin wrapper around the combine device kernel. |
| `tt_reduce.py` | `TtReduceModule` — fused weighted sum over top-k + reduce-scatter. |
| `tt_routed_expert.py` | `TtRoutedExpert` — per-expert FFN. Uses `extract`/`insert` ops to slice tokens from the flat dispatch buffer. |
| `tt_shared_expert.py` | `TtSharedExpert` — TP-sharded FFN for the shared expert. |
| `init_helpers.py` | `ExpertMapping` (expert-to-chip mapping), `compute_constants`, test input generators, mesh mapper/composer factories. |
| `tt_moe_gate_prefill.py` | `TtMoEGatePrefill` — gate matmul + grouped top-k routing + routing setup integration. |
| `tt_moe_intermediates.py` | `TtMoEIntermediates` dataclass for debugging. |
