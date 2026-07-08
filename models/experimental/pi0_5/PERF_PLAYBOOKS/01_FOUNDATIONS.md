# 01 · Foundations — Hardware, Memory, and Precision

Read this before any component file. Every "flew in isolation, crashed in the model"
outcome reduces to one of the limits below being silently broken.

---

## 1. The core grid — never hard-code it

| Device | Compute grid | Cores | DRAM banks | L1 / core |
|---|---|---:|---:|---:|
| Wormhole (n150/n300) | 8 × 8 | 64 | 6 | ~1.5 MB |
| Blackhole (P150) | up to 11 × 10 / 10 × 12 | 110–120 | 8 | 1.57 MB (1,572,864 B) |

```python
# ALWAYS:
grid = device.compute_with_storage_grid_size()
core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
```

**Why it matters (all three campaigns):**
- BGE-M3: a sweep harness hard-coded to 8×8 on Blackhole produced "winning" matmul
  configs that were ~5 µs/call *slower* than `ttnn.linear`'s default 11×10 routing.
  The harness tuned against 64 cores; production already used 110.
- ViT-BH hard-codes 10×12 for max perf but documents the portable
  `compute_with_storage_grid_size()` alternative.
- Swin-L (WH) is locked to 8×8; all its matmul program configs assume 64 cores.

Corollary: `per_core_M = ceil(M_tiles / grid_y)` produces *different* per-core work on
WH vs BH for the same shape. **Tuning results do not port across architectures.**

---

## 2. L1 budget and the CB-clash equation

For a 2D-mcast matmul on grid `(gx, gy)`, the per-core L1 requirement is roughly:

```
L1_per_core ≈
    2 · in0_block_w · per_core_M · tile_size     (in0, double-buffered)
  + 2 · in0_block_w · per_core_N · tile_size     (in1, double-buffered)
  + per_core_M · per_core_N · tile_size          (output CB)
  + per_core_M · per_core_N · 4   IF fp32_dest_acc_en=True   (fp32 intermediate)
  + any resident L1 tensor on that core (sharded input, previous op's output)
```

When the sum exceeds the L1 max (1,572,864 B on BH) you get:

```
TT_THROW ... Statically allocated circular buffers in program N clash with
L1 buffers on core range [...]. L1 buffer allocated at A and static CB region ends at B.
```

The overlap is `B − A` bytes. The clash is the #1 reason a standalone sweep winner
crashes in-model: in isolation L1 is empty, in-model the previous op left a tensor resident.

**Tile sizes** (memorize):
- `bfloat16` = 2,048 B
- `bfloat8_b` = 1,088 B
- `bfloat4_b` ≈ 548 B

**Fixes when you clash** (in order):
1. Switch the offending matmul to `ttnn.experimental.minimal_matmul` (streaming-K, much smaller CB footprint).
2. Shrink the largest CB term: `in0_block_w`, then subblock, then `per_core_*`.
3. Set `fp32_dest_acc_en=False` if it was True (removes the fp32 intermediate, halves DST).
4. Move the upstream tensor to DRAM (last resort — the DRAM round-trip usually costs more than the matmul win at large batch).

---

## 3. The DST register file and the subblock cap

All math output lands in DST before the packer moves it to a CB. The hardware cap on
`out_subblock_h · out_subblock_w` depends on accumulation precision:

| `fp32_dest_acc_en` | Subblock cap | Effect |
|---|---:|---|
| `True` (fp32 accumulate) | `h·w ≤ 4` | half DST capacity |
| **`False`** (fp16 accumulate) | **`h·w ≤ 8`** | full DST capacity |

This is the **single highest-leverage matmul knob.** Flipping to False unlocks subblocks
like `1×8`, `2×4`, `8×1` that are physically illegal at True. Doubling subblock area
roughly halves pack/unpack round-trips.

**Decision rule for every matmul:** try `fp32_dest_acc_en=False` first → run full-model
PCC → re-sweep subblock with the new `h·w ≤ 8` ceiling. Only fall back to True if PCC
fails. (Normalization reductions and sometimes softmax need True; bf8b matmul almost
never does.)

---

## 4. `packer_l1_acc` — must match production in any sweep

`packer_l1_acc=True` accumulates matmul output directly in the L1 CB instead of a
DST→L1 round-trip per tile. It is the production default for matmul/SDPA/norm on both
architectures.

**A sweep harness with `packer_l1_acc=False` reports matmul times ~3.5× slower than
reality** and produces bogus "winners." This was a real multi-day bug in the BGE-M3
campaign (Sweep 4.1). Always:

```python
compute = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.LoFi,   # per-op, see §5
    math_approx_mode=False,
    fp32_dest_acc_en=False,                  # per-op, see §3
    packer_l1_acc=True,                      # ALWAYS in sweeps
)
```

---

## 5. Math fidelity — the precision/throughput dial

| Fidelity | Mantissa bits / mul | Math iters | Relative cost | Use for |
|---|---:|---:|---:|---|
| **LoFi** | 5 | 1 | 1× | bf8b matmul, bf8b SDPA, GELU LUT |
| **HiFi2** | 7 | 2 | 2× | bf16 matmul, **normalization**, mixed-precision SDPA |
| HiFi3 | 8 | 3 | 3× | rare; almost always HiFi2 or HiFi4 wins |
| HiFi4 | 10 | 4 | 4× | first-pass safety; almost never needed at bf8b |

**Empirical fidelity policy that held across campaigns:**

| Op | Fidelity | Note |
|---|---|---|
| bf8b matmul (QKV, MLP, attn-out) | **LoFi** | 4× cheaper than HiFi4; PCC holds |
| bf16 matmul | HiFi2 | LoFi at bf16 gave no win in practice |
| **LayerNorm / GroupNorm / RMSNorm** | **HiFi2 + fp32_dest=True** | LoFi here compounds to failure over depth |
| SDPA at bf8b | LoFi | with `fp32_dest_acc_en=True` for softmax sum |
| SDPA at bf16 / high-precision ViT | HiFi2 or HiFi4 | ViT high-res uses HiFi4 for score precision |

**Throughput multipliers (vs HiFi4):** the Tensix math engine runs ~2x faster at HiFi2 and
~3.6x faster at LoFi. This matters most for *compute-bound* matmuls (prefill); for
*DRAM-bound* matmuls (decode) the weight read dominates and fidelity barely moves the needle.

**Fidelity-by-weight-dtype rule (LLM convention):**
| Weights | Fidelity | Note |
|---|---|---|
| bf16 | HiFi4 | or when accuracy is critical (often attention) |
| bf8b | HiFi2 | drops the LSB of a bf16 @ bf8b mul; usually fine. LoFi often also works |
| bf4b | LoFi | matches the 4-bit mantissa; works for many MLP matmuls |

ViT-BH note: it uses `math_approx_mode=True` for its sharded LayerNorm compute config
(the approximate rsqrt is safe at ViT's 12-layer depth and 224 seq). BGE-M3 at 24 layers
keeps `math_approx_mode=False` for LN. **Depth matters** - see section 7.

---

## 6. Memory configs — the five that matter

| Config | Where | Use for |
|---|---|---|
| `DRAM_MEMORY_CONFIG` | DRAM, interleaved | large tensors that don't fit L1; large-batch activations; weights |
| `L1_MEMORY_CONFIG` | L1, interleaved | small tensors used by adjacent ops |
| block-sharded L1 | L1, grid_y splits M / grid_x splits N | LayerNorm, 2D-mcast matmul I/O — **the ViT default for the whole encoder** |
| height-sharded L1 | grid splits M only | attention BMMs (per-head), 1D-mcast matmul |
| width-sharded L1 | grid splits N only | 1D-mcast inputs, decode-style activations |
| DRAM-sharded weights | DRAM, per-bank N-slices | decode (small M, weight-bandwidth-bound) — **almost never for prefill** |

**Matmul variant follows the regime** (full detail in 08): Matmul 2D for prefill/encoder
(compute-bound, DRAM interleaved); DRAM-sharded for decode (bandwidth-bound, L1
width-sharded act); `minimal_matmul` as the L1-CB escape hatch when 2D mcast clashes.

**The reshard rule:** every transition between memory configs is a real op (1–100 µs).
Match producer→consumer layouts to avoid them. But a reshard the next op performs
*internally anyway* (e.g. some matmuls re-shard their in0) is not yours to remove — test.

**Always fuse dtype casts into reshards:**
```python
# Bad: two ops
y = ttnn.typecast(x, ttnn.bfloat16); z = ttnn.interleaved_to_sharded(y, cfg)
# Good: one op
z = ttnn.interleaved_to_sharded(x, cfg, output_dtype=ttnn.bfloat16)
```

---

## 7. The depth-compounding principle (why single-layer PCC lies)

Reduction ops (LayerNorm mean/var, softmax sum) accumulate per-call quantization error.
Over an N-layer encoder this compounds **multiplicatively**:

| Change | 1 layer PCC | 24-layer PCC |
|---|---:|---:|
| LoFi LayerNorm | 1.0000 | **0.91 — FAIL** |
| one-pass LN reduction | 0.999 | **0.89 — FAIL** |
| bf8b sharded LN (no bf16 cast) | 0.9999 | **0.50 — FAIL** |
| fp32_dest=False on LN | 0.999 | **0.89 — FAIL** |

Matmul/SDPA compounding is far milder; single-layer PCC is usually a reliable predictor
there. **For any reduction op, always validate at full model depth before shipping.**
A 12-layer ViT tolerates approximations a 24-layer BERT does not.

---

## 8. Regime decision — pick your playbook

Measure first:
```
host_overhead = wall_time − device_time          # device_time = Σ signpost-bounded kernel ns
host_fraction = host_overhead / wall_time
```

| Condition | Regime | Primary levers |
|---|---|---|
| `host_fraction > 30%` | **host-bound** (small batch, decode, fast device) | trace capture, 2-CQ, op-count reduction, fuse ops |
| `host_fraction < 5%` | **device-bound** (large batch, long seq) | fidelity walk, subblock unlock, remove typecasts, L1 handoffs |
| activation fits in 64-core L1 | **L1-resident** (ViT, BGE-M3 B1) | block-shard the whole encoder, sharded LN, no DRAM round-trips |
| activation ≫ L1 | **DRAM-streaming** (BGE-M3 B32, high-res ViT) | DRAM interleaved default, selective L1 handoffs, DRAM-stage Q/K/V for SDPA |

The same model at different batch sizes needs different strategies — see each component
file's per-regime sections.

---

## 9. Quick reference

| Knob | Default | When to change |
|---|---|---|
| Core grid | `device.compute_with_storage_grid_size()` | never hard-code |
| `math_fidelity` | LoFi (bf8b matmul/SDPA), HiFi2 (norm/bf16) | walk down per op, PCC-gated |
| `fp32_dest_acc_en` | False (matmul), True (norm) | try False first on matmul |
| `packer_l1_acc` | True | never False in a sweep |
| `math_approx_mode` | False | True only if the shape/depth proves it safe |
| Activation dtype | bf8b | keep native through attention; don't cast to bf16 |
| Weights | DRAM interleaved | DRAM-sharded only for decode |
| Reshards | minimize | fuse dtype cast; don't remove ones the next op redoes |
