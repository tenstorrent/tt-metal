# Sharding & Data-Flow Reference: qwen3.6 v2 vs llama3_70b_galaxy

Both models target an 8×4 Galaxy mesh (32 chips). Mesh shape `(8, 4)` is
hardcoded in multiple sites (`llama70b/llama_ccl.py:1255`, `lm_head.py:114`,
qwen36 default in `qwen36_model_config.py:240`). The semantic mapping is:

- `cluster_axis=0` → mesh dim 0 (rows = 8) → the **8-way ring**
- `cluster_axis=1` → mesh dim 1 (cols = 4) → the **4-way ring**

---

## 0. Row vs column semantics — what each axis is FOR

A 2D mesh gives you two orthogonal axes to distribute work. The mesh **doesn't
care** which axis you call "rows" vs "cols" — that's a convention. What matters
is **what kind of parallelism each axis carries**:

```
                    ┌──────────────────────────────────────────┐
                    │   8 rows × 4 cols = 32 chips             │
   ┌──────────┐     │                                            │
   │ axis = 0 │     │      col0   col1   col2   col3            │
   │  rows=8  │ ──► │  row0 [ ] [ ] [ ] [ ]                     │
   │ 8-way    │     │  row1 [ ] [ ] [ ] [ ]                     │
   │ ring     │     │  row2 [ ] [ ] [ ] [ ]                     │
   └──────────┘     │  row3 [ ] [ ] [ ] [ ]    ◄── cluster      │
                    │  row4 [ ] [ ] [ ] [ ]        _axis=0 ring │
                    │  row5 [ ] [ ] [ ] [ ]        traverses    │
                    │  row6 [ ] [ ] [ ] [ ]        a column     │
                    │  row7 [ ] [ ] [ ] [ ]        (8 chips)    │
                    │                                            │
                    │              ◄── cluster_axis=1 ring ──►   │
                    │              traverses a row (4 chips)     │
                    └──────────────────────────────────────────┘
```

Both models choose to encode parallelism kinds on these axes as follows:

### Llama70b: **2D Tensor Parallelism (2D-TP)**

Llama70b spreads **distinct slices of the model** across every chip — no two
chips do identical compute. The 32 chips are arranged as:

- **Rows (cluster_axis=0, 8-way)** carry the **head dimension** parallelism.
  64 Q heads / 8 rows = 8 Q heads per chip. Different rows compute different
  attention heads.
- **Cols (cluster_axis=1, 4-way)** carry the **hidden dimension** parallelism.
  hidden=8192 / 4 cols = 2048 hidden-dim per chip. Different cols hold
  different slices of the residual stream.

Each chip is a UNIQUE compute participant. The 32 chips collaborate, none
duplicates another's work.

```
WQKV weight layout (per llama_attention.py line 202):
  ttnn.ShardTensor2dMesh(dims=(3, 2), mesh_shape=(8, 4))
                              │  │
                              │  └─ tensor dim 2 (hidden=8192) on cols (4) → 2048 per chip
                              └──── tensor dim 3 (qkv_size=10240) on rows (8) → 1280 per chip
  Per-chip WQKV: [1, 1, 2048, 1280]
```

After matmul + `llama_rs_create_heads(cluster_axis=1)`:
- The reduce-scatter on cols (4-way) completes the hidden-dim contraction
- Heads are already row-distributed (8 per chip from row sharding)
- Per chip: 8 Q heads + 1 K head + 1 V head, all locally complete

After SDPA, WO matmul: `dims=(2, 3)` → input (head*hd=8192) on rows, output
(hidden=8192) on cols. Output partial sum reduces across rows (the 8-way ring,
`line_all_reduce(cluster_axis=0)` — the perf win).

### Qwen3.6 v2: **1D Tensor Parallelism + Data Parallelism**

Qwen3.6 v2 only spreads the model across cols. The 8 rows hold **identical
copies** of the entire compute. Specifically:

- **Cols (cluster_axis=1, 4-way)** carry **all** of the model parallelism:
  - 24 Q heads / 4 cols = 6 Q heads per chip
  - 4 KV heads / 4 cols = 1 KV head per chip
  - hidden=5120: **NOT split** — full H per chip
- **Rows (cluster_axis=0, 8-way)** are **data-parallel replicas**. Every chip
  in a column does the SAME computation. The 8 rows produce 8 identical
  copies of every intermediate.

```
WQKVG weight layout (per llama_attention.py line 416):
  ttnn.ShardTensor2dMesh(dims=(None, 3), mesh_shape=(8, 4))
                              │     │
                              │     └─ tensor dim 3 (qkvg_size=14336) on cols (4) → 3584 per chip
                              └─────── None: tensor NOT split on rows → REPLICATE
  Per-chip WQKVG: [1, 1, hidden=5120, 3584]
```

After matmul (no reduce-scatter needed — hidden dim is REPLICATED not split):
- 4 cols each have a different 3584-wide slice of QKVG (6 Q + 6 Gate + 1 K + 1 V × 256)
- 8 rows within a col hold IDENTICAL output

After SDPA, WO matmul: `dims=(None, 2)` → input (n_q*hd=6144) on cols (split
4-way), output (hidden=5120) REPLICATED across both rows AND cols. The WO
partial-sum reduce-scatters on cluster_axis=1 (cols, 4-way) OR an all_reduce on
cluster_axis=0 (rows, 8-way) depending on the V2-CCL path:

- V2-CCL path: `line_all_reduce(cluster_axis=0)` on rows — but the rows are
  REPLICATED so this is summing 8 identical values × 8 → multiplying by 8.
  This is mathematically incorrect for a TP reduction!
- Actually the way it works in v2: the WO matmul output uses `cluster_axis=1`
  reduce (4-way ring) for the actual sum, then any cluster_axis=0 op is
  reducing already-replicated values which is just a no-op multiplier baked
  into the residual contract.

The bottom line: **qwen3.6 v2 throws away the 8-way row dimension for
attention** — uses it purely as data-parallel replication. Llama70b uses it as
the head-split dimension.

### Why the divergence?

The split is forced by head-count divisibility:

| | llama70b | qwen3.6 v2 |
|---|---|---|
| n_q_heads | 64 | 24 |
| n_kv_heads | **8** | **4** |
| n_q % 8 (rows) | 0 ✓ | 0 ✓ (would be 3/chip) |
| n_kv % 8 (rows) | 0 ✓ | **4/8 = 0.5 ✗** |
| n_q % 4 (cols) | 0 ✓ (16/chip) | 0 ✓ (6/chip) |
| n_kv % 4 (cols) | 0 ✓ (2/chip) | 0 ✓ (1/chip) |

For llama70b, BOTH axes divide cleanly into both Q and KV head counts. So
2D-TP works trivially. For qwen3.6 v2, **`n_kv_heads=4` doesn't divide by 8
rows**. To shard KV heads on rows would require KV head replication (4 → 8)
which doubles cache memory.

The simplest workaround chosen at qwen3.6 v2 design time: skip the
row-sharding of heads entirely. Put everything on cols (where 4 divides into
both 24 and 4), and replicate across rows. This avoids the KV pad refactor but
sacrifices 8× compute efficiency (rows duplicate work) and gives up the 8-way
ring as a head-reduction axis.

### Consequences of qwen3.6 v2's choice

1. **Row-replicated compute on full-attention weights.** Each col's 8 rows
   compute the same QKVG, SDPA, WO. The 8-way ring becomes a data-parallel
   replication axis instead of a TP-reduction axis — for full-attn only.

2. **WO reduce ring is 4-way, not 8-way** for the natural cluster_axis=1 reduce.
   Per-call WO RS latency is ~2× what llama70b achieves. (V2-CCL moved one path
   to cluster_axis=0 but as noted that path is mathematically a no-op-multiply.)

3. **Residual stream is full-H per chip** (5120 elements per chip) instead of
   H/4 (1280). 4× the residual memory footprint per chip.

4. **Two extra all_gather/mesh_partition pairs per layer** to convert between
   col-sharded (norm input/MLP output) and full-H (attention input/residual
   add). See section 5.

5. **DeltaNet escape hatch**: DeltaNet's separate head counts (16 K / 48 V,
   both divisible by 8) allow it to use `cluster_axis=0` for its head split
   without divisibility issues. This is why DeltaNet sharding is asymmetric
   to full_attn in qwen3.6 v2 — DeltaNet uses the 8-way ring properly, full_attn
   doesn't.

### Important nuance: row-replication is NOT uniform across the model

The qwen3.6 v2 sharding choice replicates on rows for SOME weights but not all.
Audit by weight:

| weight | shard dims | rows replicate? | reason |
|---|---|---|---|
| **Full-attn WQKVG** (`llama_attention.py:416`) | `(None, 3)` | **YES** | col-only; 1D-TP, 8× redundant compute |
| **Full-attn WO** (`llama_attention.py:440`) | `(None, 2)` | **YES** | col-only; 1D-TP, 8× redundant compute |
| **Full-attn KV cache** (`llama_attention.py:775`) | `(None, 1)` | **YES** | n_kv split on cols only; rows hold identical KV |
| **DistributedNorm `weight_distributed`** | `(None, 2)` | **YES** | tiny weight (5120 floats); intentional |
| **MLP w1, w3** (`llama_mlp.py:96-104`) | `(-1, -2) = (3, 2)` | **NO** | 2D-TP — intermediate on rows, hidden on cols |
| **MLP w2** (`llama_mlp.py:97-103`) | `(-2, -1) = (2, 3)` | **NO** | 2D-TP — intermediate on rows, hidden on cols |
| **DeltaNet weights** | custom row-based | **NO** | n_v_per_row = 48/8 = 6 heads per row |
| **DistributedNorm `weight`** (full-H replica) | `ReplicateTensorToMesh` | YES (replicated everywhere) | tiny; replicated for fast access |
| **Embedding** | (loaded as replicated typically) | YES | small lookup |
| **LM head** | sharded on output vocab dim | NO | vocab-parallel |

**So the row-replication problem is SPECIFIC to full-attention.** MLP is properly
2D-sharded across rows AND cols — both axes carry distinct parallelism. DeltaNet
likewise splits across rows. The 8× redundancy ONLY hits the 16 full-attention
layers (of 64 total).

This refines the cost picture:
- 16 layers × 4 redundant ops (QKVG + WO + 2 matmul-internal) = ~64 redundant matmuls per forward
- 48 layers × 0 redundant = DeltaNet is fine
- 64 layers × 0 redundant = MLP is fine
- Total per-forward redundant work ≈ 16/64 = **25% of attention/MLP compute is row-redundant**

Not 8× the entire model — closer to 8× of 25% = effectively 2-3× the full-attn
attention compute that COULD be parallelized but isn't.

### Why was MLP done correctly and attention wasn't?

Looking at git history:
- MLP code is inherited mostly verbatim from `llama3_70b_galaxy/tt/llama_mlp.py`,
  which uses 2D-TP. Qwen3.6 just changed the intermediate size and dtype.
- Attention code (the qwen36 branch in `llama_attention.py`) was a NEW path
  written for qwen3.6's specific structure (QKVG fusion, per-head QK-norm,
  partial RoPE, output gate). This code chose the simpler 1D col-sharded layout
  rather than re-deriving llama70b's 2D-sharding for qwen3.6's head counts.

So MLP inherits llama70b's good 2D-TP "for free", while attention was
reinvented and ended up with the simpler-but-worse 1D layout.

This explains why fixing only attention would recover most of the benefit:
the MLP is already correct. The 2D-TP refactor proposal in section 5b is
specifically about converting full-attention to 2D-TP. MLP doesn't need touching.

### Could v2 use llama70b's 2D-TP? Yes, with KV padding (4 → 8)

The blocker is the 4 KV heads not dividing into 8 rows. The fix is to replicate
each KV head once at weight-load time (`k_w.repeat_interleave(2, dim=0)`),
making n_kv = 8 (4 unpadded × 2). This is exactly the "olmo-style pad" we
explored earlier. After padding, qwen3.6 has effectively 8 KV heads matching
llama70b's structure and can use 2D-TP with all the same benefits.

The cost: 2× KV cache memory (still small in absolute terms — 5120/8 = 640
elements per chip per cache slot). Manageable.

This is the "real" path to bring qwen3.6 v2's compute efficiency in line with
llama70b, IF we can also solve the precision floor that's currently capping
64L decode PCC at 0.30 regardless of sharding.

---

## 1. Head counts & per-chip distribution

| | llama70b | qwen3.6 v2 |
|---|---|---|
| `n_q_heads` | 64 | 24 |
| `n_kv_heads` | 8 | 4 |
| `head_dim` | 128 | 256 |
| total Q*hd | 8192 | 6144 |
| total KV*hd | 1024 | 1024 |
| `hidden_size` | 8192 | 5120 |

**Head split axis: `cluster_axis=1` (cols=4) for BOTH models.**

| | llama70b | qwen3.6 v2 |
|---|---|---|
| n_local_heads (Q/col) | 64/4 = **16** | 24/4 = **6** |
| n_local_kv_heads (KV/col) | 8/4 = **2** | 4/4 = **1** |
| Q per chip | 16 | 6 |
| KV per chip | 2 | 1 |
| Q*hd per chip | 2048 | 1536 |

Cleanly divisible in both — no head padding required.

---

## 2. DeltaNet (linear_attention) sharding — qwen3.6 v2 only

DeltaNet has its own head set, sharded on **`cluster_axis=0` (rows=8)**:

| | qwen3.6 v2 DeltaNet |
|---|---|
| `linear_num_key_heads` | 16 |
| `linear_num_value_heads` | 48 |
| `linear_head_dim` | 128 |
| n_k_per_row | 16/8 = **2** |
| n_v_per_row | 48/8 = **6** |
| Q heads per row | 16/8 = 2 (GQA expanded from K) |

DeltaNet uses an **asymmetric mesh axis**: heads on rows (cluster_axis=0), data parallel on cols (cluster_axis=1 replicates → 4 cols hold identical copies).

---

## 3. Weight sharding

### llama70b (2D-sharded, residual is col-sharded)

```python
# WQKV weight: [hidden=8192, qkv_size=10240]
self.wqkv = ttnn.as_tensor(
    pt_wqkv,
    mesh_mapper=ttnn.ShardTensor2dMesh(
        dims=(3, 2),  # dim 3 on rows (8), dim 2 on cols (4)
        mesh_shape=(8, 4),
    ),
)
# Per chip: [hidden/4=2048, qkv_size/8=1280]

# WO weight: [n_q*hd=8192, hidden=8192]
self.wo = ttnn.as_tensor(
    pt_wo,
    mesh_mapper=ttnn.ShardTensor2dMesh(
        dims=(2, 3),  # dim 2 on rows (8), dim 3 on cols (4)
        mesh_shape=(8, 4),
    ),
)
# Per chip: [n_q*hd/8=1024, hidden/4=2048]
```

### qwen3.6 v2 (1D-col-sharded, residual is full-H replicated)

```python
# WQKVG weight: [hidden=5120, qkvg_size=14336]
self.wqkvg = ttnn.as_tensor(
    wqkvg_T,
    mesh_mapper=ttnn.ShardTensor2dMesh(
        dims=(None, 3),  # only dim 3 split (on cols=4); rows REPLICATE
        mesh_shape=(8, 4),
    ),
)
# Per chip: [hidden=5120, qkvg_size/4=3584]

# WO weight: [n_q*hd=6144, hidden=5120]
self.wo = ttnn.as_tensor(
    wo_T,
    mesh_mapper=ttnn.ShardTensor2dMesh(
        dims=(None, 2),  # only dim 2 split (on cols=4); rows REPLICATE
        mesh_shape=(8, 4),
    ),
)
# Per chip: [n_q*hd/4=1536, hidden=5120]
```

**Key difference**: llama70b is 2D-sharded; qwen3.6 v2 is 1D-col-sharded with row replication. This makes the qwen3.6 residual stream **full-H per chip** vs llama70b's **H/4 per chip**.

---

## 4. KV cache layout

### llama70b
```python
# Per chip cache shape: [batch/4, n_kv_heads/4=2, max_S, head_dim=128]
mesh_mapper=ttnn.ShardTensor2dMesh(dims=(None, 1), mesh_shape=(8, 4))
# dim 1 (n_kv_heads=8) sharded on cols → 2 KV per chip
# rows replicate
```

### qwen3.6 v2
```python
# Per chip cache shape: [B, n_kv_heads/4=1, max_S, head_dim=256]
mesh_mapper=col_shard_kv  # similar pattern: dim 1 on cols
# rows replicate
```

Same pattern — KV heads split on cols, rows replicate. Per-chip footprint scales with hidden/4.

---

## 5. Residual stream — the BIG difference

### llama70b (col-sharded throughout)

```
[Layer N output: col-sharded H/4 per chip, rows replicate]
        │
        ▼
   norm(x_sharded)                ← DistributedNorm consumes col-shard,
        │                             produces col-shard
        ▼
   attention(x_norm_sharded):
        │
        ├─ matmul → xqkv_fused_sharded (col-sharded)
        ├─ llama_rs_create_heads(cluster_axis=1)
        │   ← REDUCE-SCATTER on cols (4-way), produces row-sharded heads
        ├─ RoPE, SDPA (per-chip heads)
        ├─ WO matmul → dense_partial (col-sharded H/4)
        └─ line_all_reduce(cluster_axis=0)
            ← REDUCE on rows (8-way, the BIG ring)
            output stays col-sharded H/4
        │
        ▼
   residual_add (x_sharded + attn_out_sharded) → col-sharded
        │
        ▼
   ff_norm (col-sharded → col-sharded)
        │
        ▼
   MLP:
        ├─ FF1/FF3 matmul → reduce_scatter(cluster_axis=1)
        ├─ all_gather between → SwiGLU
        ├─ FF2 matmul → line_all_reduce(cluster_axis=0)  ← 8-way ring
        ▼
   residual_add → col-sharded H/4 (input to next layer)
```

**Inter-layer: col-sharded H/4. No mesh_partition / all_gather between blocks.**

### qwen3.6 v2 (full-H replicated, with internal scatter+gather around norm)

```
[Layer N output: full-H replicated per chip (5120 per chip), all cols identical]
        │
        ▼
   mesh_partition(cluster_axis=1)        ← FREE — selects col-shard view
        │                                    (per chip: H/4 = 1280)
        ▼
   DistributedNorm(x_col_sharded)        ← bf16-stat all_gather across cols
        │                                    (the precision-floor concern)
        ▼
   attn_in_sharded (col-sharded)
        │
        ▼
   all_gather(dim=3, cluster_axis=1)     ← REAL all_gather, restores full-H
        │                                    (5120 per chip)
        ▼
   attention(attn_in_full):
        │
        ├─ matmul x @ wqkvg → xqkvg (per chip: [B, T, total_per_col=3584])
        │   (wqkvg is row-replicated, col-sharded; matmul output col-sharded)
        ├─ Q/Gate/K/V slicing per col (3584 = 6 Q heads × 256 hd × 2 + ...)
        ├─ QK-norm, partial RoPE, SDPA (per-chip, with GQA expand n_q/n_kv=6)
        ├─ Output gate: σ(Gate) * attn_out
        ├─ WO matmul → dense_partial (per chip: [B, T, H=5120], col-sharded × 4)
        └─ line_all_reduce(cluster_axis=0) or ttnn.all_reduce(cluster_axis=1)
            ← in current code, cluster_axis=1 (4-way ring) on default path
            ← cluster_axis=0 (8-way ring) on V2-CCL env-gated path
            output: full-H replicated
        │
        ▼
   residual_add (x_full + attn_out_full) → full-H replicated
        │
        ▼
   mesh_partition(cluster_axis=1)        ← FREE — re-slice for ff_norm input
        │
        ▼
   ff_norm (col-sharded → col-sharded)
        │
        ▼
   _mlp_decode_qwen36 (col-sharded → col-sharded):
        ├─ FF1/FF3 → line_reduce_scatter(cluster_axis=1)
        ├─ all_gather between → SwiGLU
        ├─ FF2 → line_all_reduce(cluster_axis=0)  ← 8-way ring
        ▼
   all_gather(cluster_axis=1)            ← gathers MLP output back to full-H
        │                                    for residual add against full-H h_new
        ▼
   residual_add (h_new + ff_out_full) → full-H replicated
        │
        ▼
   mesh_partition(cluster_axis=1)        ← exit scatter for next layer
        │
        ▼
[next layer input: col-sharded H/4 per chip]
```

**Inter-layer: col-sharded H/4 entry, full-H during compute, col-sharded H/4 exit.**

The qwen3.6 v2 pattern essentially does the same thing as llama70b but with EXTRA mesh_partition + all_gather pairs around each block to convert col-sharded ↔ full-H. This was an early-architecture choice to fit qwen3.6-specific block shapes; it adds CCL ops but is structurally simpler than re-deriving llama70b's 2D sharding for qwen3.6's head counts.

---

## 5b. The "extra all_gathers" — full deep-dive on the tradeoff

### What's actually different per layer

The qwen3.6 v2 decoder adds two inter-block CCL ops that llama70b doesn't have:

| step | llama70b | qwen3.6 v2 |
|---|---|---|
| **norm input** | col-sharded (from prior layer's output) — free | col-sharded via `mesh_partition` (free view) |
| **norm output** | col-sharded — feeds attn directly | col-sharded → **`all_gather(cluster_axis=1)`** → full-H |
| attention | consumes col-sharded H/4 | consumes full-H 5120 |
| attention out | col-sharded H/4 (via WO 2D shard + line_all_reduce on rows) | full-H (via WO 1D col-shard + all_reduce on cols) |
| residual add 1 | col-sharded + col-sharded | full-H + full-H |
| ff_norm | col-sharded (input was already col-sharded) | full-H → **`mesh_partition`** (free) → col-sharded |
| MLP | col-sharded → col-sharded | col-sharded → col-sharded |
| MLP out | col-sharded — feeds residual add directly | col-sharded → **`all_gather(cluster_axis=1)`** → full-H |
| residual add 2 | col-sharded + col-sharded | full-H + full-H |
| exit | col-sharded (next layer input) | full-H → **`mesh_partition`** (free) → col-sharded |

**Net per-block diff:**
- llama70b: 0 extra CCL ops
- qwen3.6 v2: **2 extra all_gathers on `cluster_axis=1` (4-way ring)** + 2 mesh_partitions (free)

### Concrete cost of the extra all_gathers

Each `all_gather` on cluster_axis=1 (cols, 4-way) brings the missing 3/4 of the residual into each chip:

| | decode T=1 | prefill T=128 |
|---|---|---|
| Residual tensor size | 5120 × 2 bytes = 10 KB | 5120 × 128 × 2 = 1.31 MB |
| Bytes moved per all_gather | ~7.5 KB (3/4 of tensor) | ~983 KB |
| 2 all_gathers × 64 layers | **1.28 MB / step** | **168 MB / prefill** |
| Per-call latency (BH, 4-way ring, num_links=1) | ~30-50 µs | ~200-400 µs |
| 2 × 64 = 128 calls per forward | **4-6 ms / decode step** | **25-50 ms / prefill** |

For decode at ~17 tok/s (~58 ms/step), the extra 4-6 ms is ~7-10% of the step
budget — non-trivial. For prefill, the extra 25-50 ms is moderate.

### Why v2 chose this anyway

1. **Block-internal simplicity.** DeltaNet and full-attention (with QKVG fusion,
   per-head QK-norm, partial RoPE, output gate) were written assuming **full H is
   visible per chip**. Their math operates on full hidden vectors. Converting
   them to consume col-sharded H/4 inputs requires:
   - Per-chip Q/K/V/Gate slicing logic to handle col-shards correctly
   - QK-norm per-head with col-sharded stats (cross-device sync inside the norm)
   - Partial RoPE applied to col-sharded slices (already-distributed cos/sin tables)
   - WO matmul with 2D-sharded weight (requires the KV-pad refactor to make heads divisible by rows=8)

2. **Avoided the KV-pad refactor.** Llama70b's col-sharded contract pairs with
   2D-TP weight sharding. 2D-TP requires `n_kv % 8 == 0`. Qwen3.6 has 4 KV
   heads; padding to 8 was deemed too invasive at design time.

3. **Was "good enough" at small layer counts.** At 4L or 16L, the extra
   all_gathers are a few ms total — not a regression worth restructuring for.

### What "ideal" would look like

The ideal sharding for qwen3.6 on an 8×4 Galaxy mesh is **llama70b's 2D-TP
pattern, with KV heads padded 4 → 8**. Concretely:

| dimension | qwen3.6 v2 current | qwen3.6 ideal (= llama70b pattern + KV pad) |
|---|---|---|
| n_kv_heads | 4 (native) | **8 (padded via `repeat_interleave(2, dim=0)`)** |
| Head split axis | cols (4-way) | **rows (8-way)** |
| Hidden split axis | not split (rows replicate) | **cols (4-way)** |
| Q per chip | 6 (24/4) | **3 (24/8)** |
| KV per chip | 1 (4/4) | **1 (8/8)** |
| Hidden per chip | 5120 (full) | **1280 (5120/4)** |
| WQKVG shard | `dims=(None, 3)` 1D | **`dims=(3, 2)` 2D** |
| WO shard | `dims=(None, 2)` 1D | **`dims=(2, 3)` 2D** |
| WO reduce axis | cols (4-way) | **rows (8-way) — 2× faster ring** |
| Residual stream | full-H, replicated | **col-sharded H/4** |
| Inter-block CCL ops | 6 (with 2 extra all_gathers) | **4** |
| Per-chip compute redundancy | 8× (rows duplicate) | **1× (no redundancy)** |
| KV cache memory | 1× | **2× (KV pad doubles entries)** |
| Code complexity (blocks) | medium (full-H math) | high (col-sharded math + sub-head ops) |
| Code complexity (decoder) | high (extra layout ops) | low (uniform col-sharded throughout) |

### Quantified benefits of the ideal

1. **Eliminate 2 all_gathers per layer.** Decode: ~5 ms/step saved. Prefill:
   ~35 ms saved. At 17 tok/s baseline, this is +1-2 tok/s on decode.

2. **8-way ring on WO/FF2 reduce.** Per-call WO RS latency drops from ~80 µs
   (4-way) to ~40 µs (8-way). Across 16 full-attn layers × 1 WO + 64 layers ×
   1 FF2 = 80 ops × 40 µs = 3.2 ms saved per decode step. Another +1 tok/s.

3. **8× compute utilization.** Currently 8 rows along a col duplicate work.
   With 2D-TP, those 8 rows compute different heads. ATTENTION + MLP throughput
   could nominally 8× — but this is bounded by other factors (memory bandwidth,
   CCL ring). Realistic improvement: 2-4× per attention/MLP throughput.

4. **4× less residual memory per chip.** 5120 → 1280 floats per chip. Frees
   L1 for other buffers (helps with the L1-CB-collision issues we hit during
   the single-chip-norm experiment).

5. **Simpler decoder code.** No mesh_partition/all_gather wrappers around
   every norm/MLP. Cleaner to trace, easier to debug.

### Costs of the ideal

1. **KV cache 2×.** With 8 padded KV heads instead of 4, per-chip cache
   doubles in the KV-head dimension. For decode at ISL=4k, B=32 this is
   ~64 MB → 128 MB per chip per layer × 16 full-attn layers = ~2 GB total
   per chip. Fits but uses headroom.

2. **2-3 weeks of refactor work.** Touches: weight loading, attention forward,
   decoder forward, KV cache init, paged update / SDPA, CCL buffer keys,
   per-op program configs. Per-block PCC tests need to be redone.

3. **GQA grouping math.** With 4 → 8 KV padding via `repeat_interleave(2,0)`,
   GQA group size goes from 6 (24/4) to 3 (24/8). The per-Q-head mapping to
   KV index changes from `q//6` to `q//3`. Pre-flight CPU test verified the
   replication preserves attention math bit-identically.

4. **DeltaNet stays unchanged.** DeltaNet's sharding is already on rows
   (16 K / 48 V both divisible by 8). Only full-attention needs the refactor.

### Why we shouldn't pursue the ideal RIGHT NOW

The previous V2-2D commits started this refactor. They got most of the way
through (config + weight loader + forward) before hitting:

1. **A test bug** (`test_layer3_full_attention_forward_pcc.py`) that gave
   misleading PCC=0.7075 results. This was a measurement issue, not a real
   regression.

2. **The actual precision bug** (the same one we've been chasing) was
   conflated with the refactor. With unit tests now proving pure bf16 noise
   gives PCC > 0.999 at 64L, we know the v2 PCC issue (0.30) is a
   **separate, deeper bug** unrelated to sharding architecture.

The 2D-TP refactor would deliver perf gains (+2-3 tok/s, ~10% throughput) but
not fix the 64L decode PCC issue. So the order should be:

1. **First**: find and fix the precision bug capping 64L decode at 0.30
2. **Then**: do the 2D-TP refactor to recover the 2× WO ring + eliminate the
   inter-block all_gathers

Doing them in the opposite order risks the same conflation we hit during V2-2D.

---

## 6. CCL ops summary — where data crosses chips

| op | llama70b (cluster_axis) | qwen3.6 v2 (cluster_axis) |
|---|---|---|
| DistributedNorm stats all_gather | 1 (cols, 4-way) | 1 (cols, 4-way) |
| QKVG `llama_rs_create_heads` | 1 (cols, 4-way) | not used — col-sharded mm + per-col slicing |
| Norm input gather (qwen36-specific) | n/a | 1 (cols, 4-way) ← inserted by decoder |
| WO `line_all_reduce` | 0 (rows, **8-way**) | 0 or 1 (env-gated; default=0 post V2-CCL) |
| MLP FF1/FF3 `reduce_scatter` | 1 (cols, 4-way) | 1 (cols, 4-way) |
| MLP intermediate `all_gather` | 1 (cols, 4-way) | 1 (cols, 4-way) |
| MLP FF2 `line_all_reduce` | 0 (rows, **8-way**) | 0 (rows, **8-way**) |
| Inter-block residual gather | none | 1 (cols, 4-way) ← MLP-output gather |
| Inter-block residual partition | none | mesh_partition (free) |

**Net per-block CCL count:**
- llama70b: 1 RS on cols + 1 AG on cols (in MLP) + 2 AllReduce on rows (WO + FF2) = 4 CCL ops
- qwen3.6 v2: same 4 core CCL ops + 2 extra all_gather (norm input + MLP output) + 2 mesh_partition (free) = 6 CCL ops

The 2 extra all_gathers per layer are the qwen3.6 v2 overhead vs llama70b.

---

## 7. Distributed RMSNorm internals (shared between both models)

Both call `tt_distributed_rmsnorm` (prefill) / `tt_sharded_distributed_rmsnorm` (decode) from `llama70b/llama_ccl.py`.

```python
def tt_distributed_rmsnorm(inp, epsilon, gamma, ...):
    # inp: col-sharded [B, T, H/4]

    # Step 1: per-chip sum-of-squares stat
    tt_stats = ttnn.rms_norm_pre_all_gather(
        inp, dtype=ttnn.bfloat16,   # ← bf16-quantized stat
    )
    # Step 2: gather stats across cols (4-way)
    tt_stats_gathered = line_all_gather(tt_stats, cluster_axis=1, buffer_key="LAYERNORM")
    # Step 3: kernel-asserted bf16 stats → post_all_gather normalize
    tt_out = ttnn.rms_norm_post_all_gather(
        inp, tt_stats_gathered, weight=gamma,
    )
    # rms_norm_post_all_gather kernel HARD-ASSERTS stats.dtype ∈ {bf16, bf8}
    # — fp32 stats are REJECTED at the C++ level
    return tt_out
```

The bf16 stats precision floor IS enforced by the C++ kernel, but per-unit-test results, this floor compounds to only ~4% rel-err over 64L. **Not enough alone to explain the observed 70% drift.**

---

## 8. Reference files

- `models/demos/llama3_70b_galaxy/tt/llama_attention.py` — canonical attention (full-attention)
- `models/demos/llama3_70b_galaxy/tt/llama_mlp.py` — canonical MLP
- `models/demos/llama3_70b_galaxy/tt/llama_decoder.py` — canonical decoder block
- `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1255` — hardcoded `mesh_shape=(8, 4)`
- `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1297` — `tt_distributed_rmsnorm` (prefill)
- `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1332` — `tt_sharded_distributed_rmsnorm` (decode)
- `models/demos/qwen3_6_galaxy_v2/tt/qwen36_model_config.py:240` — qwen36 cluster_shape default
- `models/demos/qwen3_6_galaxy_v2/tt/llama_attention.py:409` — qwen36 WQKVG weight loader (1D)
- `models/demos/qwen3_6_galaxy_v2/tt/llama_attention.py:433` — qwen36 WO weight loader (1D)
- `models/demos/qwen3_6_galaxy_v2/tt/llama_decoder.py:354-498` — qwen36 decoder forward (full-H residual contract)
- `models/demos/qwen3_6_galaxy_v2/tt/qwen36_delta_attention.py` — DeltaNet block
- `models/demos/qwen3_6_galaxy_v2/tt/distributed_norm.py` — DistributedNorm wrapper (uses llama70b's primitives)

---

## 9. Known precision floor (V2-DEC investigation)

64L decode PCC = 0.30 on qwen3.6 v2. Pure bf16 compounding (per unit tests) gives PCC > 0.999 at 64L. The 0.30 must come from:
- A sub-bf16 precision somewhere (e.g., bf8b in some intermediate buffer)
- An algorithm-level math mismatch (RoPE / SDPA / DeltaNet decay)
- A specific kernel with numerical bug

NOT from bf16 noise in the residual stream, norm stats, matmul outputs, or residual adds.
See `BRINGUP_LOG.md` V2-DEC entries for the full investigation chain.
