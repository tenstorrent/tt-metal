# 01 · Foundations — Hardware, Memory, and Precision

Read this before any component file. Every "flew in isolation, crashed in the model"
outcome reduces to one of the limits below being silently broken.

---

## 1. The core grid — never hard-code it {#fnd-core-grid}
<!-- route
grid: partial,tiny
lever_type: single-shot
-->

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

---

## 10. Shard the activation into L1 — the coordinated edit {#shard-activation-to-l1}
<!-- route
op_class: matmul,datamove
memory: dram_interleaved
lever_type: structural
-->

**Fires when:** the bucket is `memory=dram_interleaved`. That tag means every op reads
its inputs from DRAM — the activation is never resident in L1, so the matmul stalls on the
DRAM read and the chip streams the same data back and forth (this is also the root of a
large `datamove` op count). This is the **highest-leverage structural win** on an encoder
whose activation fits in 64-core L1.

**THE #1 FAILURE MODE — read this first.** Setting a matmul `program_config` (block sizes,
core grid) on an input tensor that is still `DRAM_INTERLEAVED` **does NOTHING** — the kernel
graph is unchanged and the edit is inert. A program config is *not* a sharding op. You MUST
change the **`memory_config` of the tensor itself**. Sharding is a property of the *tensor*,
not the matmul call.

**The edit is a COORDINATED change — all three parts, or it's a no-op:**

1. **Shard the input activation into L1** *before* the matmul:
   ```python
   grid = device.compute_with_storage_grid_size()          # full grid — never hard-code
   shard_cfg = ttnn.create_sharded_memory_config(
       shape=x.shape,                                       # (B·S, H)
       core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
       strategy=ttnn.ShardStrategy.BLOCK,                   # block-sharded (encoder default)
       orientation=ttnn.ShardOrientation.ROW_MAJOR,
   )
   x = ttnn.to_memory_config(x, shard_cfg)                  # <-- THE op that actually shards
   ```
2. **Give the matmul the matching program config + full grid** (so the math runs on all cores):
   ```python
   ttnn.linear(x, w, bias=b,
       program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
           compute_with_storage_grid_size=(grid.x, grid.y),
           in0_block_w=H_tiles // grid.x,
           out_subblock_h=1, out_subblock_w=min(4, N_tiles // grid.x),
           per_core_M=ceil(M_tiles / grid.y), per_core_N=ceil(N_tiles / grid.x),
           transpose_mcast=False, fused_activation=<existing activation or None>),
       memory_config=shard_cfg,                              # keep OUTPUT sharded too
       compute_kernel_config=<existing HiFi2/LoFi config>)
   ```
3. **Keep the output sharded** so the *next* op (LN / residual / next matmul) consumes it
   without a reshard back to DRAM. Chaining the shard down the layer is what collapses the
   `datamove` count — a single sharded matmul whose output immediately reshards to DRAM is a
   wash.

**FORCE-TRY (do not skip as "risky"):** make the full coordinated change and let the PCC gate
decide. Do NOT ship a partial version (e.g. program-config only) to be "safe" — a partial
shard is exactly the inert no-op above. If L1 clashes (the CB equation in §2), shrink
`in0_block_w` / subblock or switch to `ttnn.experimental.minimal_matmul` — but still shard
the tensor. The only acceptable reason to fall back to DRAM is a *measured* PCC failure or a
*measured* L1 overflow, never caution.

---

## 11. Occupy the full core grid {#occupy-full-grid}
<!-- route
op_class: matmul,datamove,reduction
grid: partial,tiny
lever_type: structural
-->

**Fires when:** the bucket is `grid=partial` (or `tiny`) — the ops ran on fewer than all 64
(WH) / 110+ (BH) cores, so most of the chip is idle. Almost always a *consequence* of an
unsharded, DRAM-interleaved tensor: with no shard spec, the op gets a default/partial core
assignment.

**The fix is the same coordinated shard as §10** — you raise occupancy by *sharding the
tensor across the full grid*, not by passing a bigger `compute_with_storage_grid_size`
alone (that, by itself, is the inert no-op). Concretely:

1. `grid = device.compute_with_storage_grid_size()` — resolve the real grid, never hard-code.
2. Shard the input across `(grid.x, grid.y)` (see §10 step 1).
3. Set the program config's `compute_with_storage_grid_size=(grid.x, grid.y)` and size
   `per_core_M / per_core_N` so the work divides evenly across all cores.

**FORCE-TRY:** attempt full-grid occupancy on the hottest op in the bucket; PCC-gate it.
Verify the result by re-profiling — the bucket's `grid` tag should flip from `partial` to
`full`. If it stays `partial`, the tensor was not actually sharded (the §10 failure mode) —
fix the `memory_config`, don't just retune the program config.

---

## 12. Mixed-precision experiment — mandatory, don't skip as risky {#mixed-precision-experiment}
<!-- route
op_class: matmul,reduction
fidelity: hifi4,hifi2
lever_type: structural
-->

**Fires when:** the bucket runs at `fidelity=hifi4` (max precision, 4× the math cost) or a
`hifi2` matmul whose weights are still bf16. High fidelity is the safe *bring-up* default,
not a *perf* choice — the Tensix math engine runs ~2× faster at HiFi2 and ~3.6× faster at
LoFi (see §5). A model brought up for correctness almost always leaves precision headroom on
the table. **The "Make Fast Models Fast" lesson is explicit: skills that don't MANDATE a
mixed-precision experiment leave this win unclaimed ("BFP8 MLP weights ... did not mandate
mixed-precision tests").**

"Mixed precision" = different ops at different precisions, each walked down to the lowest its
PCC tolerates — NOT one global dtype. The coordinated change per matmul:

1. **Weights → lower dtype where safe:** bf16 → `bfloat8_b` (often) → `bfloat4_b` (many MLP
   matmuls). Cast the weight tensor's dtype, don't just change a kwarg.
2. **Match the compute kernel fidelity (§5 policy):**
   ```python
   compute = ttnn.init_device_compute_kernel_config(
       device.arch(),
       math_fidelity=ttnn.MathFidelity.LoFi,    # bf8b/bf4b matmul; HiFi2 for bf16
       math_approx_mode=False,
       fp32_dest_acc_en=False,                   # matmul; unlocks h·w≤8 subblocks (§3)
       packer_l1_acc=True,
   )
   ```
3. **Pass it to the op** (`compute_kernel_config=compute`).

**The PCC-aware policy — this is what keeps it safe (§5/§7):**
| Op | Try | Hard floor (do NOT cross) |
|---|---|---|
| bf8b matmul (QKV/MLP/attn-out) | **LoFi** | — usually holds |
| bf16 matmul | HiFi2 | LoFi rarely wins at bf16 |
| **LayerNorm/RMSNorm/softmax reductions** | stay **HiFi2 + fp32_dest=True** | **NEVER LoFi** — compounds to FAIL over depth (§7) |

So for the `reduction` bucket: drop HiFi4 → HiFi2 on *non-normalization* reductions, but keep
norms at HiFi2 + fp32. For `matmul`: walk weights down + LoFi, PCC-gate each step.

**FORCE-TRY (the deck's exact instruction):** make the precision drop and let the **full-model**
PCC gate decide — do NOT skip it as "risky." Precision wins are the most commonly-skipped
optimization because they *feel* risky; the whole point is the gate makes trying them free.

**SELF-VERIFY:** after editing, confirm the op's `compute_kernel_config` fidelity actually
changed (and/or the weight dtype) — a fidelity drop is a real kernel change (unlike a bare
program_config), so this one should move the device-time number, not just the op graph.
**Validate reductions at FULL MODEL DEPTH** (§7) — single-layer PCC lies for norms.

## 13. Activation dtype walk — shrink bytes MOVED on the memory-bound path {#dtype-walk-activations}
<!-- route
op_class: datamove,eltwise,matmul
lever_type: structural
-->

**Complements §12 (do not confuse):** §12 (`mixed-precision-experiment`) walks *matmul/reduction
WEIGHT* dtype + compute fidelity. THIS lever walks the *ACTIVATION* tensors that flow *between*
ops — the data that `datamove` ops (reshape/transpose/reshard/tilize) physically move and that
`eltwise`/`matmul` consume. On a **datamove-dominant** model (Mamba/SSD, conv-as-shift-select,
heavy reshapes) the biggest bucket is movement, and movement time = **bytes moved**, so halving
the activation data-format ~halves it. §12 never touches this path.

**Fires when:** a `datamove` or `eltwise` bucket holds significant time, or a `matmul`/op the
roofline tags `bound_by=memory` (bytes/DRAM-bw floor dominates, not FLOPs). The roofline's
per-op `bytes` + `bound_by=memory` is the signal that *fewer bytes* — not more cores — is the win.

**The change (a TENSOR data-format change, not a kwarg):**
1. **Carry activations in a smaller format between ops:** bf16 → `bfloat8_b` (usually safe) →
   `bfloat4_b` (where the gap is large and PCC tolerates). Set it on the *producing* op's output
   dtype / via `ttnn.to_memory_config`/`ttnn.typecast` so the tensor that gets reshaped/transposed/
   resharded is physically smaller. Each step halves (bf16→bf8b ~1.9×, →bf4b ~3.5×) the bytes the
   datamove op moves.
2. **Remove redundant up-casts:** if an op up-casts to bf16/fp32 only to be moved then down-cast,
   keep it low through the movement.
3. **Walk DOWN per tensor, PCC-gate each step:** start bf8b; if the full-model PCC holds and the
   gap is still large, try bf4b; if PCC fails, REPAIR_PCC backs off one step. Lowest PCC-safe wins.

**Hard floors (do NOT cross — §7):** never push **normalization** activations or **KV-cache /
attention scores** below bf8b (compounds to FAIL over depth); never bf4b a tensor feeding a
softmax/reduction. Weights are §12's job, not this lever's.

**FORCE-TRY:** make the activation down-cast and let the **full-model** PCC gate decide — this is
the most-skipped win on memory-bound models. The gate makes trying it free.

**SELF-VERIFY (this one is measurable):** after the edit, the `datamove`/affected bucket's **`bytes`
and device-ms must drop** (the roofline captures per-op bytes) — if bytes are unchanged you only
changed a kwarg, not the tensor's stored format; fix it. Validate at FULL MODEL DEPTH (§7).
