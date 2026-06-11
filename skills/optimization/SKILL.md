---
name: performance-optimization
description: Optimize TTNN models for maximum throughput using metal tracing, memory optimization, op fusion, and L1 sharding. Use when optimizing model performance, implementing tracing, fusing ops, or targeting decode latency.
---

# SKILL: Performance Optimization

## Scope: per-block

This skill is **per-block**. It tunes one TTNN module at a time:
compute kernel config (HiFi4 + fp32_dest_acc), memory layout (DRAM
TILE), weight dtype, sharding individual matmuls, fusing
block-internal sequences.

For **pipeline-level perf** — cross-block refactors like
`paged_update_cache` migration, reusable metal trace across
`generate()` calls, integrated tracy on the full AR pipeline — see
`skills/perf/SKILL.md`. Those patterns require touching multiple
block files at once and don't fit per-block dispatch.

A leaf block already using HiFi4 + fp32_dest_acc + bf16 DRAM TILE is
at-ceiling for this skill. "No improvement found → status=ok" is a
valid outcome.

## Purpose
Optimize TTNN models for throughput using a profiler-driven loop: **measure → bucket → attack → verify**. Cover tracing, sharded ops, memory placement, and op fusion.

## Profiler-driven workflow (always start here)

Don't guess where time goes — measure. Build a short profiler harness for the model (one warmup + traced replay) and run it under tracy. Then bucket the CSV by op-code and by `memory_config.memory_layout` / `buffer_type` to see L1 vs DRAM split.

### Tracy harness pattern

```python
# tt/profile_<model>.py
import argparse, ttnn, torch
from your.model import YourTtModel

parser = argparse.ArgumentParser()
parser.add_argument("--traced", action="store_true")
args = parser.parse_args()

dev = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=50_000_000)
m = YourTtModel(...)

# Warmup compiles all kernels into the program cache
for _ in range(3):
    out = m.forward(inputs)
ttnn.synchronize_device(dev)

if args.traced:
    m.capture_trace(...)
    m.execute_trace(...)   # one warmup replay
    ttnn.synchronize_device(dev)

# Profiled iteration -- one call so the CSV is small
out = m.execute_trace(...) if args.traced else m.forward(inputs)
ttnn.synchronize_device(dev)
ttnn.close_device(dev)
```

Run:
```
python3 -m tracy -p -v -r --op-support-count 50000 \
  models/path/tt/profile_<model>.py --traced
```

CSV lands in `generated/profiler/reports/<TIMESTAMP>/ops_perf_results_*.csv`.

### Aggregation snippet

```python
import csv, collections, re
rows = list(csv.DictReader(open(path)))
sess = [r for r in rows if r.get("METAL TRACE REPLAY SESSION ID","").strip() == "1"]

# By op type
buckets = collections.Counter()
counts = collections.Counter()
total = 0
for r in sess:
    ns = int(r["DEVICE KERNEL DURATION [ns]"] or 0)
    if ns:
        buckets[r["OP CODE"].strip()] += ns
        counts[r["OP CODE"].strip()] += 1
        total += ns

# By output memory layout (find what's DRAM vs L1)
mc = collections.Counter()
for r in sess:
    ns = int(r["DEVICE KERNEL DURATION [ns]"] or 0)
    if not ns: continue
    m = re.search(
        r"'(?:output_memory_config|memory_config|output_mem_config)':"
        r"\s*'MemoryConfig\(memory_layout=TensorMemoryLayout::(\w+);buffer_type=BufferType::(\w+)",
        r["ATTRIBUTES"]
    )
    mc[(m.group(1)+"/"+m.group(2)) if m else "?"] += ns
```

What to look for in the output:
- Total device kernel time per trace replay should match measured latency (within ~5% launch overhead).
- **DRAM share is the optimization budget**. Anything > 10% interleaved/DRAM is worth attacking.
- **Single-core ops are red flags** — `CORE COUNT == 1` for what should be a parallelizable op (LN, matmul, head split, concat) means the kernel fell back to its single-core path because the input wasn't sharded.
- Group `BinaryNgDeviceOperation` by `binary_op_type` and memory_config: a 20× gap between DRAM (~80 µs/op) and L1 (~4 µs/op) is common, and exposes the missing `memory_config=L1` hint.

### First ask: can this hot op be REPLACED, not just tuned?

Before tuning a hot op's memory_config/sharding, ask whether it should exist at
all. The most common case: a hot `ReshapeView` + `Slice` + `Transpose` chain
that is a hand-written **QKV head split** (or head merge). The fix is NOT to
L1-pin the reshape — it is to **replace the whole chain with the fused
primitive** `ttnn.experimental.nlp_create_qkv_heads` /
`nlp_concat_heads` (reshape input to 4D first). Measured: a head-split chain at
~38% of attention device-kernel time → fused op cut the block ~40%; L1-pinning
the reshape instead only bought ~24% and left the wasteful op in place. Tuning a
suboptimal op is a half-fix; eliminating it is the win. Other replace-don't-tune
cases: materialized `[1,nh,seq,seq]` softmax → flash/windowed SDPA; manual
`repeat` before multiply → broadcast multiply; `layer_norm`+`linear` → fused.

### Transformer attention prefill — standard fused primitives (common across models)

These recur in every decoder attention block; apply them by default rather than
transliterating the HF reference op-by-op. Cross-check against
`models/tt_transformers/tt/attention.py::forward_prefill` — the canonical pattern.

- **Fused RoPE**: replace the hand-written `mul(cos) + mul(rotate_half(x), sin)`
  chain (rotate_half itself is slice+slice+neg+concat → 6 ops per tensor, ×2 for
  Q/K = 12 ops/layer) with `ttnn.experimental.rotary_embedding(x, cos, sin)`
  (cos/sin shaped `[1,1,seq,head_dim]`, input bf16). One kernel per tensor.
  Note: the op tile-pads seq to a multiple of 32, so pad/realign tensors that
  bypass it (e.g. V) and slice the SDPA output back to the logical seq.
- **Native GQA in SDPA**: `ttnn.transformer.scaled_dot_product_attention` accepts
  Q=`[1,nh,S,hd]` with K/V=`[1,nkv,S,hd]` when `nh % nkv == 0` — it expands KV
  internally. **Drop the manual `_repeat_kv`** (reshape+`repeat`+`reshape`, often
  +tilize = ~4 ops/layer). Pass the nkv-headed K/V straight in.
- **SDPA chunk size**: the prefill `SDPAProgramConfig` `q_chunk_size`/`k_chunk_size`
  default low (128). For seq ≥ ~2048, **q256/k512 is ~2× faster** and still fits L1
  (512/512 overflows the CB → TT_THROW). Mirror tt_transformers'
  `get_attn_sdpa_prefill_program_config` (256 if seq≥2048 else 64). Keep the
  device's real grid (`compute_with_storage_grid_size()`), NOT a hardcoded (8,8)
  — that constant is n150-specific; Blackhole/p150 is (13,10). This applies to
  the VLM **vision** windowed SDPA too (measured dots.ocr vision: 46→25 ms/layer
  at q256/k512, seq=19520).

### VLM vision attention — bf8 Q/K/V + HiFi2 SDPA (the qwen2.5-VL pattern)

A ViT/VLM vision encoder runs **full bidirectional attention over all patches in
every layer** (no causal mask, often no window unless the config has
`window_size`/`fullatt_block_indexes` — dots.ocr has neither, so all 42 layers are
full O(seq²) over ~19.5k patches). You can't shrink the attention scope without
changing the architecture, so attack the SDPA kernel precision: **typecast Q/K/V to
`bfloat8_b` and run the SDPA with a HiFi2 / `fp32_dest_acc_en=False` compute config**
(separate from the HiFi4 config used for the matmuls/RoPE). Vision attention
tolerates this — measured dots.ocr vision SDPA **46 → 15 ms/layer (3.1×)** at
seq=19520, and the **full-tower PCC stayed 0.99998** (attention block 0.99975).
qwen2.5-VL vision (`models/demos/qwen25_vl/tt/vision_attention.py`) does exactly
this. ALWAYS gate on the full-tower PCC test (error accumulates across 42 layers)
+ e2e token-match, not just a single-block PCC. The matmuls/RoPE stay HiFi4/bf16.

## Memory hierarchy + sharding (the biggest lever)

### The default kills you
Most `ttnn` ops, when called without explicit `memory_config`, output to **DRAM interleaved**. The next op then reads from DRAM. Even ops you think landed in L1 (sharded conv, sharded matmul) **coalesce their final output back to DRAM-interleaved if `memory_config` isn't passed**. Always pass `memory_config=` explicitly.

Worse: this also degrades parallelism. `ttnn.rms_norm` / `ttnn.layer_norm` / `nlp_create_qkv_heads` / `nlp_concat_heads` **fall back to single-core kernels** when their input isn't sharded. A single op at 1 core / ~20 µs becomes 64 cores / ~3 µs the moment the input is properly sharded.

### Conv outputs land DRAM-interleaved by default
`ttnn.conv1d` / `ttnn.conv_transpose2d` slice the activation across DRAM internally (you'll see HEIGHT_SHARDED per-slice in the profile), then **coalesce the final tensor back to DRAM-interleaved**. The downstream op reads from DRAM unless you either:
- Pass `memory_config=ttnn.L1_MEMORY_CONFIG` to the conv (works for tile-aligned shapes; may regress when L1 budget contended with weights), or
- Pass `shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED` in `Conv1dConfig` (requires channel count divisible by grid_x with per-col tile-aligned; only works for C ≥ 256 in {tile-multiple} sets), or
- Explicitly `interleaved_to_sharded(...)` immediately after the conv.

### Sharded LN / RMSNorm (engages multi-core path)

Default `ttnn.rms_norm(x_l1_interleaved)` → single-core kernel.

Fix: width-shard the input across N cores (block_w=1 tile per core) and pass `LayerNormShardedMultiCoreProgramConfig`. Mirror the helper at `models/demos/inworld_tts/tt/mlp.py:_build_sharded_norm_pc`:

```python
def _build_sharded_norm_pc(phys_h_tiles, K_tiles, grid_x, grid_y):
    num_cores = grid_x * grid_y
    block_w = max(1, K_tiles // num_cores)
    subblock_w = min(4, block_w)
    while block_w % subblock_w != 0:
        subblock_w -= 1
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        subblock_w=subblock_w,
        block_h=phys_h_tiles,
        block_w=block_w,
        inplace=False,
    )
```

Then shard the input width-wise and call LN with `program_config=...` and `memory_config=L1_WIDTH_SHARDED_MEMORY_CONFIG`. Measured impact: ~17 µs single-core → ~2 µs / 32 cores (−88%) for hidden_dim=1024 in qwen3_tts CP.

### Sharded matmul / linear (2D block-sharded)

For a `[M, K] @ [K, N]` matmul where M_tiles, K_tiles, N_tiles all divide cleanly across an (gx, gy) core grid (`K_tiles % gx == 0`, `N_tiles % gx == 0`, `M_tiles % gy == 0`), use the helpers in `models/demos/inworld_tts/tt/mlp.py`:

```python
from models.demos.inworld_tts.tt.mlp import (
    _pick_grid, _block_shard, _build_block_sharded_pc
)

M_phys = 1
for d in range(len(x.shape) - 1):
    M_phys *= x.padded_shape[d]
M_t, K_t, N_t = M_phys // 32, x.shape[-1] // 32, w.shape[-1] // 32
gx, gy = _pick_grid(M_t, K_t, N_t, max_x=8, max_y=8)
if gx >= 2 and M_t % gy == 0 and K_t % gx == 0 and N_t % gx == 0:
    per_M, per_K, per_N = M_t // gy, K_t // gx, N_t // gx
    x_bs = _block_shard(x, per_M, per_K, gx, gy)
    pc = _build_block_sharded_pc(per_M, per_N, per_K, gx, gy, activation=None)
    out = ttnn.linear(
        x_bs, w, bias=b,
        program_config=pc,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=compute_cfg,
    )
else:
    # Fall back to core_grid + interleaved
    out = ttnn.linear(x, w, bias=b, core_grid=ttnn.CoreGrid(y=gy, x=gx),
                     memory_config=L1, compute_kernel_config=compute_cfg)
```

Wrap this in a `_sharded_linear(x, w, b, ...)` helper and use it everywhere (W2V QKV, attention out_proj, FFN linears, fusion linear). Chain through SiLU sharded too — elementwise ops preserve sharded layout transparently.

### 1D width-sharded matmul (for tall/skinny shapes)

When M_tiles is small (e.g. M_tiles=1 in decode, or per-call activations with only one tile of height), use `MatmulMultiCoreReuseMultiCast1D` instead of 2D block-sharded. Shard the activation on K across the compute grid, return to L1 interleaved at block boundaries. Pattern from inworld_tts vocos: 82 µs → 41 µs (−50%) for `fc_post_a` at K=2048.

### Width-sharded vs 2D block-sharded: pick by M_tiles
- **M_tiles == 1 (decode, single-token)**: 1D width-sharded. Forces `in0_block_w=1` but you can't parallelize on M anyway.
- **M_tiles >= 4 (prefill, small bucket)**: 2D block-sharded. Splits M across `grid_y` rows, raising `in0_block_w = K_tiles/grid_x`. Measured (inworld_tts MLP): fc1 K=1024 N=4096 went 118 → 75 µs (−37%) by switching from 1D to 2D at M_tiles≥4.
- **M_tiles large (e.g. batch=64 prefill)**: gate back to `core_grid` interleaved — `_USE_1D_M_CAP=16` style cutoff — because sharded paths exceed per-core L1 budget.

### Generalize sharded gates beyond `seq_len == 1`
A common bug: gating the sharded fast path on `is_decode and seq_len == 1` excludes prefill seqs 2..32, which all tile-pad to M=32 and still satisfy the sharded matmul's M==tile_height constraint. **Relax to `seq_len <= 32`**. Measured (qwen3_tts CP_prefill, seq=2): 1 core × 39 µs → 16 cores × 1.5 µs (~26× faster).

### DRAM-sharded matmul (for big-K paths)

For matmuls where `K >= 2048` (e.g. attention output projection, MLP down), the DRAM-sharded variant beats L1-interleaved. Halve `num_cores` from the K-N gcd so each core gets 2 K-tiles (`in0_block_w=2`) — the kernel overlaps compute and DRAM read better:

Measured (qwen3_tts talker decode @ K=2048):
- QKV  60.96 → 38.43 µs (−37%)
- O    45.64 → 24.60 µs (−46%)
- MLP gate/up 83.37 → 54.95 µs (−34%)

This only wins for K ≥ 2048. For K=1024 (small), the L1-interleaved default beats DRAM-sharded because `K_tiles=32` caps DRAM-sharded parallelism at ≤16 cores. **Apply selectively — profile before and after.**

### Keep activations sharded across the whole block

Collapse a `matmul → sharded_to_interleaved → rms_norm → add → interleaved_to_sharded → matmul` reshard chain into a single width-sharded layout that flows from the matmul output through `rms_norm` and the residual add directly into the next matmul. Requires a sharded LN program config that matches the matmul's shard (`LayerNormShardedMultiCoreProgramConfig` with same `compute_with_storage_grid_size` and `block_w = K_tiles / num_cores`).

Apply across the whole transformer block:
- Drop `sharded_to_interleaved` after `c_proj` (attention out)
- Drop `sharded_to_interleaved` after `fc2` (MLP down)
- Pass the sharded LN config so RMS norms preserve the shard
- Reshard once at the block boundary (not after every sub-op)

### Sharded `nlp_create_qkv_heads` (KV-group-interleaved weights)

Default with L1_INTERLEAVED input → single-core. Restructure `wqkv` weight to KV-group-interleaved (kvgi) layout so the matmul output is naturally width-sharded across `num_kv_heads` cores with `shard_width = (num_q_per_kv + 2) * head_dim`. Then reshape `xqkv` to that WIDTH_SHARDED layout and feed it to `nlp_create_qkv_heads`. Engages multi-core path: 1c × 39 µs → 16c × 1.85 µs (−95%).

### Sharded `nlp_concat_heads` (1 head/core)

Default with unsharded input → single-core. Reshape `attn_out` to `HEIGHT_SHARDED` across `num_heads` cores (1 head/core, shard shape `[M, head_dim]`). Measured: 1c × 20 µs → 16c × 1.2 µs (−94%).

### HEIGHT_SHARDED for narrow-channel high-T activations

When channel count is too small to tile-align for BLOCK_SHARDED (C ∈ {1, 48, 96, 192, 384} won't divide an 8-col grid with tile-aligned per-col slice), use HEIGHT_SHARDED — distribute the T axis. Helper:

```python
def _try_height_shard_l1(x, device, max_cores=64):
    if x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return x
    B, _, T, C = x.shape
    phys_h = B * T
    TILE = 32
    if phys_h % TILE != 0:
        return x  # Need tile-aligned height.
    h_tiles = phys_h // TILE
    # Pick the largest num_cores <= max_cores that divides h_tiles cleanly.
    num_cores = next((nc for nc in range(min(max_cores, h_tiles), 0, -1)
                      if h_tiles % nc == 0), 0)
    if num_cores == 0:
        return x
    per_core_h = phys_h // num_cores
    per_core_w = ((C + TILE - 1) // TILE) * TILE
    # Build CoreRangeSet that fits the device grid, then ShardSpec, then convert.
    ...
    return ttnn.interleaved_to_sharded(x, mc)
```

Watch out for static circular-buffer L1 overflow at compile: even when the *data* fits L1 distributed, the per-op CB staging area can exceed the 1.5 MB per-core L1. Mitigate by increasing num_cores (smaller per-core slice). If it still doesn't fit, fall back to DRAM for that op only.

### Pin elementwise ops to L1 via `memory_config=L1`

`ttnn.multiply` / `ttnn.add` / `ttnn.sin` / `ttnn.silu` / `ttnn.pow` default their output to DRAM-interleaved. When chained on a hot path:
- Pass `memory_config=ttnn.L1_MEMORY_CONFIG` to every op, OR
- Inherit the input's layout: `mc = x.memory_config(); ttnn.multiply(a, x, memory_config=mc)`.

Inherit-mc is the safer pattern when upstream is sharded (you don't want to unshard for one op then re-shard). Profile after — sometimes the upstream is DRAM and inherit-mc keeps DRAM, in which case explicit L1 helps.

### ⚠️ But NEVER pin a LARGE activation to L1 — it poisons the next matmul (the inverse trap)

L1-pinning helps small/decode tensors (above). At **large prefill seq it backfires catastrophically**, and the cost lands on the *downstream matmul*, not the pinned op — so it's easy to misattribute. A `[seq, hidden]` activation is `seq × hidden × 2` bytes (bf16): at seq≈4900, hidden=1536 that's ~15 MB, far over the 1.5 MB/core L1 budget. Forcing it into L1 makes the next `ttnn.linear` (e.g. MLP `gate_up`) pick a pathological L1 program path.

Measured (dots.ocr LM prefill, seq=4891): the MLP `gate_up` matmul ran **34 ms with an L1-resident input vs 6 ms with a DRAM input** — same shape, same weights, same compute config. The whole decoder layer was 36 ms (its op-by-op components summed to only ~8 ms when timed individually); the missing ~28 ms was entirely this L1→matmul interaction. Root cause was `decoder_layer.prefill_kv` pinning the two residual `ttnn.add`s to `L1_MEMORY_CONFIG` unconditionally; that L1 output flowed through RMSNorm (inherits buffer type) into `gate_up`. Gating the residual mem on seq dropped **full prefill 1426 → 262 ms (5.4×)** with zero output change (exact vs HF).

Rule: **gate the activation/residual `memory_config` on seq**, the same threshold the matmul-heavy submodules use internally:
```python
mem = ttnn.L1_MEMORY_CONFIG if x.shape[0] <= 1024 else ttnn.DRAM_MEMORY_CONFIG
x = ttnn.add(x, attn_out, memory_config=mem)   # residual
...
x = ttnn.add(x, mlp_out, memory_config=mem)
```
Diagnosis tell: time the fused block AND its components separately with `synchronize`. If `full_block ≫ Σ(components)`, you have a layout-interaction stall (a big tensor crossing an L1/DRAM boundary into a matmul), not a slow kernel. Confirm by re-timing the suspect matmul with its real input's memory_config (L1 vs DRAM), preallocated — see [[feedback_matmul_microbench_upload_contamination]].

### Memory_config doesn't always stick

`ttnn.linear(x, w, core_grid=g, memory_config=L1, ...)` still outputs DRAM in some kernel paths (the 1D-mcast program config ignores caller's memory_config). The only reliable way to land L1 is the explicit sharded path: `program_config=block_sharded_pc, memory_config=L1_BLOCK_SHARDED_MEMORY_CONFIG`.

## Tracing (capture + replay)

### When to use
Models with high host-dispatch overhead (per-op kernel-launch cost dominates). Typical wins: 2-7× for LLM decode, up to 25× for vision blocks. **Required:** all ops must be compiled before capture; persistent input/output tensor addresses.

### Pattern
```python
device = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=50_000_000)

# Pre-allocate persistent input buffer (in DRAM or L1 depending on size)
input_pers = ttnn.from_torch(
    torch.zeros(shape, dtype=torch.float32),
    dtype=ttnn.bfloat16, device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,  # or DRAM
)

# Warmup -- compile every kernel that will run inside the trace
for _ in range(3):
    out = model.forward(input_pers, ...)
ttnn.synchronize_device(device)

# Capture
tid = ttnn.begin_trace_capture(device, cq_id=0)
out = model.forward(input_pers, ...)
ttnn.end_trace_capture(device, tid, cq_id=0)

# Replay: copy new data into the same address, execute trace
for batch in inputs:
    host_t = ttnn.from_torch(batch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(host_t, input_pers, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
```

### What breaks tracing
- Python `__getitem__` slices on ttnn tensors inside the forward — pre-slice on the public API boundary and hand a fixed-shape input to the traced body.
- `ttnn.ReadDeviceProfiler(device)` in the hot path — blocks on profiler buffer, can't be inside a trace. Gate behind an env var.
- Conditional logic that depends on data values — trace must be deterministic w.r.t. control flow.
- Deallocating then re-allocating a tensor at a different address — keep persistent buffers, copy into them.

### Trace harness skeleton (model class)
```python
def capture_trace(self, sample_inputs, cq_id=0):
    # Persistent inputs (one per arg)
    self._t_in = ttnn.from_torch(torch.zeros_like(sample_inputs),
                                  dtype=ttnn.bfloat16, device=self.device,
                                  memory_config=L1)
    # Warmup
    out = self._forward_core(self._t_in)
    if isinstance(out, ttnn.Tensor): ttnn.deallocate(out)
    ttnn.synchronize_device(self.device)
    # Capture
    self._tid = ttnn.begin_trace_capture(self.device, cq_id=cq_id)
    self._t_out = self._forward_core(self._t_in)
    ttnn.end_trace_capture(self.device, self._tid, cq_id=cq_id)

def execute_trace(self, new_input, cq_id=0):
    host = ttnn.from_torch(new_input.contiguous(),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(host, self._t_in, cq_id=cq_id)
    ttnn.execute_trace(self.device, self._tid, cq_id=cq_id, blocking=False)
    return self._t_out

def release_trace(self):
    ttnn.release_trace(self.device, self._tid)
    for attr in ("_t_in", "_t_out"):
        t = getattr(self, attr, None)
        if isinstance(t, ttnn.Tensor):
            try: ttnn.deallocate(t)
            except: pass
```

## Op fusion

### QKV fusion
```python
wqkv = torch.cat([wq, wk, wv], dim=0)  # [3*dim, in_dim]
self.wqkv = ttnn.from_torch(wqkv.T.unsqueeze(0).unsqueeze(0),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=DRAM)
def forward(self, x):
    qkv = ttnn.linear(x, self.wqkv, ...)  # one matmul vs three
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, num_kv_heads=H,
                                                       transpose_k_heads=False,
                                                       memory_config=L1)
```

### Gate-Up fusion (SwiGLU)
```python
w_gate_up = torch.cat([w_gate, w_up], dim=0)
def forward(self, x):
    gate_up = ttnn.linear(x, self.w_gate_up, ...)
    gate, up = ttnn.split(gate_up, 2, dim=-1)
    return ttnn.linear(ttnn.silu(gate) * up, self.w_down, ...)
```

### Bias fusion in matmul
`ttnn.matmul`'s post-process-bias heuristic only takes the fused path when:
- The output `memory_config` is non-DRAM (L1), AND
- `core_grid` is provided (the "user_core_coord provided" branch).

Otherwise the bias becomes a separate elementwise add. Example impact (inworld_tts FSQ project_out at K=32, N=2048): 10.99 µs → 5.05 µs (−45%).

```python
out = ttnn.linear(
    x_in_L1, w, bias=b,
    core_grid=ttnn.CoreGrid(y=8, x=8),
    memory_config=L1,
    compute_kernel_config=hifi4,
)
```

**Counter-example**: forcing L1 isn't always a win. For inworld_tts `residual_fsq` `project_out` (K=32, N=2048), the accum tensor is intentionally kept in DRAM — forcing L1 makes ttnn pick a different matmul kernel variant that doesn't fuse the bias, producing a separate BinaryNg op that costs more than the L1-read savings. **Profile both choices.**

### Broadcast multiply instead of materialized repeat
`ttnn.multiply` and other elementwise ops broadcast singleton dims natively. If you find yourself doing `ttnn.repeat(y, [1, T, 1])` before multiplying with `x` of shape `[B, T, C]`, drop the repeat:

```python
# Before: 3 ops (Untilize + Repeat + Tilize) for the repeat path
y_full = ttnn.repeat(y, [1, T, 1])  # [B, 1, C] -> [B, T, C]
out = ttnn.multiply(x, y_full)

# After: 0 extra ops, broadcast handled inside multiply
out = ttnn.multiply(x, y)  # [B, T, C] * [B, 1, C] -> [B, T, C]
```

Note: `ttnn.concat` does **not** broadcast, so if the next op concatenates instead of multiplies, you still need the repeat (or restructure to use add).

### Reorder typecast + expand (GQA repeat-interleave)
For GQA where K/V at `num_kv_heads` must expand to `num_q_heads` and also typecast bf16→fp32, the order matters:

- **Slow**: typecast fp32 first → repeat_interleave at fp32 (twice the bandwidth) → ...
- **Fast**: repeat_interleave at bf16 → typecast once on the expanded tensor

Pure reordering, same math. Measured: 8 typecasts → 4 typecasts per decode, −22 µs/layer.

## CCL patterns for decode all-reduce

### `all_gather(dim=1)` + `fast_reduce_nc` beats RS+AG at 1-row shapes

The standard Megatron row-parallel pattern is reduce-scatter → all_gather (RS+AG),
which produces a replicated output. At **decode (1-row activations)** the RS shard is
`[1, 1, 1, hidden/TP]` — tiny enough that the RS kernel setup overhead dominates.
Replace with:
```python
# Replace:  all_reduce = RS + AG
# With:     all_gather the sharded activations, then reduce on host/device
h_gathered = ttnn.all_gather(h_sharded, dim=1, num_links=1,
                              topology=ttnn.Topology.Linear)
h_reduced  = ttnn.experimental.fast_reduce_nc(h_gathered, dims=[1],
                                               output_mem_config=L1)
```
`all_gather(dim=1)` collects shards from all devices along the mesh dim;
`fast_reduce_nc` reduces across that dim in a single kernel.

Measured (dots.ocr QB, MLP all-reduce, decode): **RS+AG = 30.80 µs → AG+reduce = 24.02 µs (−22%)**.

**Async CCL is not a win at 1-row.** The semaphore setup cost for async
`reduce_scatter_minimal_async` / `all_gather_async` exceeds the pipelining benefit
at 1-token batch size. Measured: async was slower than synchronous AG+reduce at
this shape. Only switch to async CCL when the per-step kernel time is ≫ the
semaphore overhead (~several µs).

## Compute kernel config (standard recipe)

```python
ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,    # less bf16 accumulation error
    fp32_dest_acc_en=True,                      # accumulate in fp32
    packer_l1_acc=True,                         # in-tile packer accumulation
)
```

Use this for every matmul, conv, and linear. HiFi4 + fp32_dest_acc is the project-standard precision floor; without it, bf16 accumulation drifts in wide-channel paths and PCC suffers.

## Activation dtype: prefer bf16

If the activation graph is fp32 but ops underneath are bf16 (RoPE, matmul weights, SDPA), every op-boundary inserts a `TypecastDeviceOperation`. Switching activations to bf16 end-to-end removes those typecasts entirely.

Measured (qwen3_tts CP, 5 layers):
- CP_Prefill 2240 → 1989 µs (−11%, −50 ops)
- Matmul 938 → 861 µs (−8%, faster bf16 matmul)
- Binary 89 → 67 µs (−25%)

Caveat: a *partial* fp32 path (e.g. fp32 KV cache fed by bf16 K/V producers) provides no benefit — the storage dtype only matters when its producer matches.

### Norm gamma weights must stay fp32

When switching activations to bf16, keep `rms_norm`/`layer_norm` gamma (and beta)
weights in **fp32**. Converting gammas to bf16 forces the norm kernel onto an
interleaved-norm dispatch path that is measurably slower than the fp32-gamma sharded
path — gammas stay resident in cache anyway, so the dtype does not save DRAM traffic.

Measured (dots.ocr QB): bf16 gammas regressed wall time ~10% vs fp32 gammas at
the same activation dtype. Always upload: `rms_norm_weight.to(torch.float32)`.

### Safe partial change: bf16 initial upload with fp32 activations

When switching the FULL activation graph to bf16 breaks PCC (e.g. fp32
softmax is mandatory due to attention sink logits at ±3000+), the initial
embedding upload to device is still safe to change to bf16. This removes
the fp32→bf16 typecast at the very start of the compute graph with no
downstream effect because the first op (norm or linear with `dtype=fp32`)
immediately upcasts.

Pattern: in `_prefill`, change only the `from_torch` dtype:
```python
h = ttnn.from_torch(embeds, dtype=ttnn.bfloat16, ...)  # was float32
```
All subsequent activations remain fp32. This is safe when:
- The first consuming op has an explicit output `dtype=ttnn.float32`
  (norm, QKV linear), so Q/K/V are fp32 regardless of h's dtype.
- The model has a measured fp32-mandatory path (attention sink, softmax
  overflow) that blocks full bf16 activation switching.

Measured (dots.ocr prefill, seq=2912): 302 → 269 ms (−11%).

### Long-sequence prefill: bf16 flash SDPA with fp32 guard

For prefill at seq > 1024, `ttnn.transformer.scaled_dot_product_attention`
with bf16 Q/K/V is faster than the manual fp32 chain. BUT: models with
attention-sink logits (e.g. Qwen2/dots.ocr layer-0 K[0] dominates at
±3122 logits) break at short seq in bf16 softmax — the sink saturates
bf16 dynamic range and the attention distribution collapses.

Guard on sequence length:
```python
USE_FLASH_SDPA_THRESHOLD = 1024  # calibrated per model

if seq > USE_FLASH_SDPA_THRESHOLD:
    # bf16 flash SDPA — correct because sink logit is diluted across
    # many query rows; per-row max remains in bf16 range.
    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q_bf16, k_bf16, v_bf16, is_causal=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=hifi4_bf16,
    )
else:
    # fp32 explicit path — mandatory for short seq where the single
    # attention-sink row dominates softmax and overflows bf16.
    attn_out = manual_fp32_sdpa(q, k, v, mask)
```

Verify the threshold with a PCC sweep: at seq=128 bf16 SDPA will fail
(PCC ~0.5); at seq=1280 it is typically fine (PCC ~0.96). Measure at
the production seq too; the sink affects EVERY query row so PCC does
not improve with longer seq — it plateaus. Gate: PCC > 0.95 for long
seq, full e2e WER parity for the guarded short-seq path.

Measured (dots.ocr prefill, seq=2912): manual fp32 → bf16 flash SDPA,
prefill 486 → 269 ms (same commit as bf16 initial upload above).

### Decode SDPA: bf16 with Q pre-scale (attention-sink models)

At decode time a **single query row sees the full key set including attention-sink
tokens**. Unlike prefill — where the sink logit is diluted row-by-row across many
queries — the decode query always hits the full ±3122 logit (dots.ocr: layer-0
key[0] peak after `scale=1/√head_dim`), so bf16 softmax overflows regardless of
seq length. Two changes unlock bf16 decode SDPA:

**1. Pre-scale Q at init** — fold `1/sqrt(head_dim)` into the Q section of the RoPE
cos/sin tables rather than applying it inside SDPA. After RoPE, Q magnitudes are
÷sqrt(head_dim), so ±3122 → ±276 before bf16 softmax sees them. Use `scale=1.0` in
the decode SDPA call (prescaling already done):
```python
# In build_full_rope_table (once at init):
decode_q_prescale = 1.0 / math.sqrt(head_dim)
cos_table[:, : num_q_heads_per_device * head_dim] *= decode_q_prescale
sin_table[:, : num_q_heads_per_device * head_dim] *= decode_q_prescale
# Pass scale=1.0 to scaled_dot_product_attention_decode
```

**2. KV cache → bf16** — once Q is pre-scaled the logits fit bf16. Switch the KV
cache from fp32 to bf16; this also halves KV-cache DRAM traffic.

Program config (Blackhole, TP=4, GQA nkv=2 replicated to 8 per device):
```python
prog_cfg = ttnn.SDPAMultiCoreProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    q_chunk_size=num_q_heads_per_device,   # e.g. 4 (16 total ÷ TP=4)
    k_chunk_size=num_kv_heads_per_device,  # e.g. 8 (nkv=2 replicated 4×)
)
compute_cfg = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,  # HiFi2 safe after Q prescaling
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

⚠️ **Prerequisite**: `nlp_create_qkv_heads_decode` silently zeros odd head rows
when its input is bf16 DRAM-interleaved on Blackhole (NoC alignment bug — see
Pitfalls). Stage the QKV output in L1 before the heads op or this change
produces wrong answers.

Measured (dots.ocr QB TP=4): **17.57 → 10.76 ms/step** (1.63×) in one REDO
combining pre-scaled Q, bf16 KV cache, and the L1-stage heads fix.

## KV-cache update patterns

### Use `fill_cache` for multi-position writes
`ttnn.update_cache(cache, k, update_idx=0)` only writes the **first row** of `k`, even if `k.shape[2] > 1`. To write N positions in one launch, use `ttnn.fill_cache`:

```python
# Before: 4 ttnn.slice + 4 ttnn.update_cache(idx=0/1) -- 8 ops to write 2 positions
# After: 1 ttnn.fill_cache writes positions 0..k.shape[2]-1 in one launch
ttnn.fill_cache(k_cache, k, 0)
ttnn.fill_cache(v_cache, v, 0)
```

### Init-time KV cache + stable pointer
For traceable prefill, allocate the max-seq KV cache **once at init** and reuse the same buffer pointer across calls — replaces the "create zeros + concat + clone fresh max_seq KV buffer" pattern at every prefill. Cache pointer stability is required for trace replay.

### Persistent trace input buffers → L1 (small) or DRAM (large)
Trace inputs that are H2D'd per-frame and read inside the trace should sit in **L1** for small tensors (embeddings, masks, cos/sin tables a few KB each) — saves a per-call DRAM→L1 staging in the consuming kernel. Exception: `paged_fused_update_cache` requires `update_idxs` in DRAM (TT_FATAL otherwise) — leave that one in DRAM.

## Long autoregressive generation: L1 fragmentation workarounds

During long AR generation, L1 fragments over many decoder calls and a sensitive op's static circular buffer can collide with a long-lived L1 buffer (`TT_THROW: Statically allocated circular buffers ... clash with L1 buffers on core range`). Stage the offending op's input through DRAM — one ~3 µs round-trip is much cheaper than the failure:

```python
# Right before the sensitive op that's hitting the collision:
x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
out = sensitive_op(x)
```

This is per-call, not per-op-everywhere. Profile which op is colliding and stage *only* its input.

## SDPA program config (prefill)

For prefill attention, replace the manual fp32 SDPA chain with a single fused call. Explicit `SDPAProgramConfig` with `compute_with_storage_grid_size=(8,8)` and `q/k chunk = 64` gives clean parallelism. HiFi4 + L1 output keeps accumulation precise and the residual add fast.

```python
attn_output = ttnn.transformer.scaled_dot_product_attention(
    q, k, v,
    attn_mask=mask,         # additive bias
    is_causal=False,
    scale=1.0/math.sqrt(head_dim),
    memory_config=L1,
    compute_kernel_config=hifi4_with_fp32_dest_acc,
    program_config=ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=64,
        k_chunk_size=64,
    ),
)
```

Measured (qwen3_tts talker prefill): manual fp32 chain → fused SDPA, prefill 51.8 → 14.4 ms (−72%).

## On-device sampling (for autoregressive decode)

Per-frame `argmax`/multinomial sampling D2H'd over a 2048-entry vocab is the dominant cost on the server hot path (≈ several ms/frame) once dispatch is hidden by 2cq. Wire `ttnn.topk` + `ttnn.sampling` into the decode trace and bake the sampled token into a tiny `[1,1,1,32]` buffer so the AR loop D2H's a single int instead of the full vocab.

Gate with an env flag (`TT_MODEL_DEVICE_SAMPLING=1`) so host and device paths both stay available — useful for A/B-ing speaker similarity / Whisper transcript match. Pattern at `tt/utils.py:_DeviceSampler` in qwen3_tts; passes `sub_core_grids` to `ttnn.topk` + `ttnn.sampling` for fine-grained core allocation.

## Custom on-device kernels (replacing host fallbacks)

When a chain of host-fallback ops dominates (e.g. STFT, ISTFT, custom activation), it's often worth implementing a full on-device replacement. Pattern from inworld_tts `CustomIstftHead`:

1. Precompute fixed tensors (IDFT matrices, window) once at init, store in DRAM TILE_LAYOUT.
2. Replace the algorithmic chain with on-device matmul / conv_transpose1d / elementwise sequence.
3. Pin every op to L1 throughout the chain.
4. End-to-end PCC vs reference must be ≥ 0.99.

Measured: full-on-device CustomIstftHead vs CPU fallback at production shapes: 13.85× speedup, PCC 0.99952+.

## Per-stage env-gated timer

For models with many stages, build a `_StageTimer` enabled via env var. When disabled, all methods are no-ops (zero overhead). When enabled, wraps each stage with `synchronize_device` so the measured time is on-device only.

```python
_TIMER_ENV = os.environ.get("MYMODEL_TIMER", "0").lower() in ("1","true","yes")

class _StageTimer:
    def __init__(self, enabled, device):
        self.enabled, self.device = enabled, device
        self.stages = []
    @contextlib.contextmanager
    def stage(self, name):
        if not self.enabled: yield; return
        ttnn.synchronize_device(self.device)
        t0 = time.perf_counter()
        try: yield
        finally:
            ttnn.synchronize_device(self.device)
            self.stages.append((name, (time.perf_counter() - t0) * 1000))
```

Use in `forward`:
```python
with self.timer.stage("attention"):
    h = self.attn(h)
with self.timer.stage("mlp"):
    h = self.mlp(h)
```

## Pitfalls

### "Statically allocated circular buffers ... beyond max L1 size"
The op's CB staging area is bigger than 1.5 MB per core. Increase num_cores (smaller per-core slice), reduce `act_block_h_override`, or fall back to DRAM for that op.

### "Statically allocated circular buffers ... clash with L1 buffers on core range"
During long autoregressive generation, L1 fragments and a long-lived buffer collides with the next op's CB region. Stage the offending op's input through DRAM (one round-trip is much cheaper than the failure):
```python
x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
out = sensitive_op(x)
```

### `to_memory_config(x, DRAM)` is *not* always a no-op
Even if the tensor is already in DRAM-interleaved, the call may be recorded as a metadata op in the trace. Cheap, but visible in profiles. The real concern is if you accidentally call it on an L1-sharded tensor — it materializes a DRAM copy.

### Conv1d won't accept HEIGHT_SHARDED input
The depthwise conv kernel at some shapes refuses HEIGHT_SHARDED input and OOMs at compile. You'll need a `sharded_to_interleaved` before the conv. Size-gate the target: prefer L1 interleaved, fall back to DRAM for the largest tensors where the conv would OOM.

### Discrete vs continuous correctness
For models with a quantizer at the end (FSQ codes, token ids), the post-quantization "code agreement" is noisy by design — sub-millith numerical differences flip codes near decision boundaries. Always report **PCC of the pre-quantization output** alongside discrete agreement; PCC ≥ 0.99 means the encoder is structurally correct even if code-agreement is in the 85-95% range.

### `nlp_create_qkv_heads_decode` silently zeros odd head rows with bf16 DRAM input (Blackhole)

On Blackhole, `nlp_create_qkv_heads_decode` produces zeroed output for odd-indexed
head rows when its input is **bf16 DRAM-interleaved**. Root cause: bf16 sub-tile is
32 B; Blackhole's DRAM NoC alignment is 64 B; odd rows start at a 32B-misaligned
offset that the decode heads kernel does not compensate for.

**Symptom**: attention output correct for even heads (0, 2, 4…), all zeros for odd
heads (1, 3, 5…); whole-head-block PCC ~0.5.

**Fix**: stage the QKV output in L1 before calling the op:
```python
xqkv_l1 = ttnn.to_memory_config(xqkv, ttnn.L1_MEMORY_CONFIG)
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_l1, num_heads=..., num_kv_heads=..., ...
)
```
The L1 round-trip costs ~1 µs and is required any time the QKV linear output may
be bf16 DRAM-interleaved on Blackhole. Also a prerequisite for bf16 decode SDPA
(see "Decode SDPA: bf16 with Q pre-scale" above).

### Single-core kernel fallback
Any of these in the profile is a smell:
- `LayerNormDeviceOperation` with CORE_COUNT=1
- `NlpCreateHeadsDeviceOperation` / `NLPConcatHeadsDeviceOperation` with CORE_COUNT=1
- A matmul with `MatmulMultiCoreReuseMultiCast1DProgramConfig` and `output_mem_config = DRAM`

All have explicit sharded paths that just need the input pre-sharded.

### Per-Conv1d `to_memory_config(x, DRAM)`
A pre-bounce to DRAM at the entry of a wrapped `TtConv1d.forward` is almost always wrong:
- If the input is already DRAM → no-op metadata only, cheap but pointless.
- If the input is L1-sharded → materializes a real DRAM copy, then conv reads it from DRAM. **Remove it**: the conv kernel will re-shard from L1 internally when configured with the right `shard_layout`.

## Performance reference points

### LLM decode (Molmo2-8B on T3K)
| Path | Latency | Throughput |
|------|---------|------------|
| Decode (traced) | ~28 ms / token | 35.6 tok/s |
| Decode (no trace) | ~180 ms / token | 5.5 tok/s |
| **Tracing speedup** | **6-7×** | |

### Vision (Molmo2-8B)
| Path | Latency |
|------|---------|
| Vision (traced) | ~86 ms |
| Vision (no trace) | ~2150 ms |
| **Tracing speedup** | **25×** |

### TTS encoder (inworld_tts, 4.49 s audio on single WH)
| Path | Latency | L1 share |
|------|---------|----------|
| Eager forward | ~187 ms | — |
| Traced | 66.4 ms | 83% |
| **Total speedup** | **2.81×** | |

### TTS decoder (inworld_tts, b=8 T=32 on single WH)
| Path | Latency |
|------|---------|
| Traced | 5.9 ms / iter |

### TTS codec predictor (qwen3_tts CP, 5 layers)
| Optimization | CP_Prefill | Cumulative |
|--------------|------------|------------|
| Baseline (fp32 activations) | 2240 µs | — |
| + bf16 activations | 1989 µs | −11% |
| + DRAM-sharded matmul (K≥2048) | 1908 µs | −15% |
| + L1-pinned activations | 1819 µs | −19% |
| + sharded nlp_concat_heads | ~1800 µs | −20% |
| + sharded nlp_create_qkv_heads | ~1750 µs | −22% |
| + sharded LN (32 cores) | ~1700 µs | −24% |

## Optimization checklist

**Profiling (always start here):**
- [ ] Build a tracy harness (`profile_<model>.py`) for the model
- [ ] Run traced + collect ops CSV
- [ ] Bucket by op-code: where does the time live?
- [ ] Bucket by `memory_layout/buffer_type`: how much is in DRAM?
- [ ] Look for `CORE COUNT == 1` ops → missing sharding

**Tracing:**
- [ ] Persistent input/output tensors (no `from_torch(...)` in hot path)
- [ ] Warmup compiles all kernels (including KV caches)
- [ ] No `ttnn.ReadDeviceProfiler` / Python control flow / `__getitem__` slicing inside trace
- [ ] `trace_region_size` is large enough (start with 50 MB)
- [ ] Slice on the public boundary, traced body takes pre-shaped inputs

**Sharding:**
- [ ] LN / RMSNorm: width-sharded input + `LayerNormShardedMultiCoreProgramConfig`
- [ ] `nlp_create_qkv_heads`: KV-group-interleaved wqkv + width-sharded matmul output
- [ ] `nlp_concat_heads`: HEIGHT_SHARDED across num_heads
- [ ] Matmul (M_tiles==1, decode): 1D width-sharded
- [ ] Matmul (M_tiles>=4, prefill): 2D BLOCK_SHARDED with `_pick_grid` + `_block_shard` + `_build_block_sharded_pc`
- [ ] Matmul (M_tiles large, batch=64): fall back to `core_grid` interleaved via `_USE_1D_M_CAP`-style gate
- [ ] Matmul (K≥2048): DRAM-sharded with `in0_block_w=2`
- [ ] Conv: `shard_layout=BLOCK_SHARDED` where channel tile-aligns; HEIGHT_SHARDED helper otherwise
- [ ] Elementwise ops: pass `memory_config=L1` (or inherit input mc) explicitly
- [ ] Sharded chain across transformer block: matmul → LN → add → matmul stays sharded
- [ ] Sharded-path gate uses `seq_len <= 32`, not `seq_len == 1` (M tile-pads to 32)

**Op fusion:**
- [ ] QKV: single fused matmul
- [ ] Gate/Up: single fused matmul
- [ ] SDPA: fused `ttnn.transformer.scaled_dot_product_attention` with explicit `SDPAProgramConfig`
- [ ] Matmul + bias: ensure L1 output + `core_grid` (but profile — sometimes DRAM keeps a different fused-bias variant)
- [ ] Broadcast multiply: no manual `ttnn.repeat` before elementwise that already broadcasts
- [ ] Typecast + expand: do expand at smaller dtype, then typecast once

**Compute / data types:**
- [ ] HiFi4 + fp32_dest_acc + packer_l1_acc on every matmul/conv
- [ ] bf16 activations end-to-end (no mid-graph fp32 enclaves that force typecasts)
- [ ] Norm gammas (rms_norm_weight) stay fp32 even when activations go bf16
- [ ] bf16 weights; fp32 only at boundaries if PCC demands
- [ ] No *partial* fp32 path (fp32 KV cache fed by bf16 K/V is dead weight)
- [ ] Attention-sink models (|logits| > 1000 post-scale): prescale Q in RoPE tables; use bf16 decode SDPA with scale=1.0
- [ ] On Blackhole: stage `nlp_create_qkv_heads_decode` input in L1 when input is bf16 (NoC alignment bug)
- [ ] MLP decode all-reduce: try `all_gather(dim=1)` + `fast_reduce_nc` vs RS+AG (22% win measured at 1-row); skip async CCL at 1-row batch

**Memory:**
- [ ] Persistent state (KV caches, mel features, FSQ dequant tables) pre-allocated once
- [ ] KV cache initialized once at module init; pointer stable across calls
- [ ] `ttnn.fill_cache` (not `update_cache`) for multi-position writes
- [ ] Inter-stage carry-over: L1 if size fits per-core CB budget, else DRAM
- [ ] Small per-call trace inputs (embeddings, masks, cos/sin) → L1; large ones → DRAM
- [ ] Drop unconditional `to_memory_config(x, DRAM)` at op entries
- [ ] After every shape-changing op, verify output `memory_config` in the profiler

**Autoregressive / streaming:**
- [ ] On-device sampling (`ttnn.topk` + `ttnn.sampling`) for hot AR loops; D2H one int, not the full vocab
- [ ] Aggregated per-frame D2H for chain mode (one transfer per frame, not per token)
- [ ] L1 fragmentation: if a sensitive op hits CB collision in long AR, stage *only its input* through DRAM

**Verification:**
- [ ] PCC ≥ 0.99 against reference (continuous output, pre-quantization)
- [ ] Trace speedup vs untraced ≥ 1.9× (if not, dispatch isn't dominant — look for compute hotspots)
- [ ] End-to-end demo path still works (run the actual app, not just unit tests)
